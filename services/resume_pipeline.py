from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Iterable

from config import PROMPTS_DIR, CV_STRUCTURE_PATH, RESUMES_DIR
from llm.llm_modules import groqLLM, openaiLLM


def _read_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def ocr_pdf_to_markdown(pdf_path: Path, groq_client: groqLLM) -> str:
    """
    Use Groq LLaMA 4 to OCR the PDF into rich Markdown.
    """
    prompt = _read_prompt("simple_ocr.txt")
    markdown = groq_client.infer(
        prompt=prompt,
        pdfs=[str(pdf_path)],
        model=None,  # use client's default
    )
    return markdown


def markdown_to_structured_json(
    ocr_markdown: str,
    openai_reasoning: openaiLLM,
) -> Dict[str, Any]:
    """
    Convert OCR markdown into strict structured JSON using GPT-5.
    """
    template = _read_prompt("structured_json.txt")
    with CV_STRUCTURE_PATH.open("r", encoding="utf-8") as f:
        schema_text = f.read()

    prompt = (
        template.replace("<##resume_ocr##>", ocr_markdown)
        .replace("<##structured_json##>", schema_text)
    )

    raw_response = openai_reasoning.infer(
        prompt=prompt,
        reasoning_effort="medium",
        model=None,
    )

    # Extract JSON between <json>...</json>
    start = raw_response.find("<json>")
    end = raw_response.find("</json>")
    if start != -1 and end != -1 and end > start:
        json_text = raw_response[start + len("<json>") : end].strip()
    else:
        json_text = raw_response.strip()

    data = json.loads(json_text)
    return data


def _ensure_candidate_dir(candidate_id: int) -> Path:
    cdir = RESUMES_DIR / str(candidate_id)
    cdir.mkdir(parents=True, exist_ok=True)
    return cdir


def process_resumes(
    pdf_files: Iterable,
    groq_client: groqLLM,
    openai_reasoning: openaiLLM,
) -> List[Dict[str, Any]]:
    """
    Given uploaded PDFs, produce:
      - original pdf
      - OCR markdown
      - structured JSON
    and return candidate records (with in-memory resume_json for scoring).
    """
    candidates: List[Dict[str, Any]] = []

    for idx, storage in enumerate(pdf_files, start=1):
        candidate_id = idx
        cdir = _ensure_candidate_dir(candidate_id)

        # Save original PDF
        pdf_path = cdir / "resume.pdf"
        storage.save(str(pdf_path))

        # OCR to markdown via Groq
        ocr_md = ocr_pdf_to_markdown(pdf_path, groq_client)
        md_path = cdir / "resume.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write(ocr_md)

        # Markdown -> structured JSON via GPT-5
        resume_json = markdown_to_structured_json(ocr_md, openai_reasoning)
        json_path = cdir / "resume.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(resume_json, f, ensure_ascii=False, indent=2)

        # Candidate name from JSON
        name = (
            resume_json.get("candidate", {})
            .get("name", {})
            .get("full")
            or f"Candidate {candidate_id}"
        )

        candidates.append(
            {
                "id": candidate_id,
                "name": name,
                "pdf_path": os.path.relpath(pdf_path, RESUMES_DIR.parent),
                "markdown_path": os.path.relpath(md_path, RESUMES_DIR.parent),
                "json_path": os.path.relpath(json_path, RESUMES_DIR.parent),
                "resume_json": resume_json,  # keep for scoring
            }
        )

    return candidates
