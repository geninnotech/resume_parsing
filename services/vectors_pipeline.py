from __future__ import annotations

import json
from typing import Dict, Any, List, Tuple

from config import PROMPTS_DIR
from llm.llm_modules import openaiLLM
from services.state import save_state


def _read_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def generate_custom_vectors(
    job_description: str,
    default_vectors_json_text: str,
    openai_reasoning: openaiLLM,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Use GPT-5 to generate JD-specific custom vectors.
    Returns (custom_vectors_list, custom_vectors_json_text).
    """
    template = _read_prompt("generate_custom_vectors.txt")

    prompt = (
        template.replace("<##job_description##>", job_description)
        .replace("<##default_vectors##>", default_vectors_json_text)
    )

    raw = openai_reasoning.infer(
        prompt=prompt,
        reasoning_effort="medium",
    )

    custom_vectors = json.loads(raw)
    custom_vectors_json_text = json.dumps(custom_vectors, ensure_ascii=False, indent=2)
    return custom_vectors, custom_vectors_json_text


def score_candidate(
    resume_json: Dict[str, Any],
    job_description: str,
    default_vectors_json_text: str,
    custom_vectors_json_text: str,
    openai_reasoning: openaiLLM,
) -> Dict[str, Dict[str, int]]:
    """
    Use GPT-5 to score a single candidate against all vectors.
    Returns:
      {
        "default_vectors": {name: score},
        "custom_vectors": {name: score}
      }
    """
    template = _read_prompt("score_candidates.txt")

    resume_json_text = json.dumps(resume_json, ensure_ascii=False, indent=2)

    prompt = (
        template.replace("<##resume_json##>", resume_json_text)
        .replace("<##job_description##>", job_description)
        .replace("<##default_vectors##>", default_vectors_json_text)
        .replace("<##custom_vectors##>", custom_vectors_json_text)
    )

    raw = openai_reasoning.infer(
        prompt=prompt,
        reasoning_effort="medium",
    )

    data = json.loads(raw)

    default_scores = {
        item["vector_name"]: int(item["score"])
        for item in data.get("default_vectors", [])
    }
    custom_scores = {
        item["vector_name"]: int(item["score"])
        for item in data.get("custom_vectors", [])
    }

    return {
        "default_vectors": default_scores,
        "custom_vectors": custom_scores,
    }


def build_and_save_state(
    job_description: str,
    default_vectors_json_text: str,
    candidates_with_json: List[Dict[str, Any]],
    openai_reasoning: openaiLLM,
) -> Dict[str, Any]:
    """
    Generate custom vectors, score all candidates, and persist a single state.json.
    """
    # 1) Custom vectors
    custom_vectors, custom_vectors_json_text = generate_custom_vectors(
        job_description=job_description,
        default_vectors_json_text=default_vectors_json_text,
        openai_reasoning=openai_reasoning,
    )

    # 2) Score candidates
    enriched_candidates: List[Dict[str, Any]] = []
    for c in candidates_with_json:
        scores = score_candidate(
            resume_json=c["resume_json"],
            job_description=job_description,
            default_vectors_json_text=default_vectors_json_text,
            custom_vectors_json_text=custom_vectors_json_text,
            openai_reasoning=openai_reasoning,
        )
        # Don't store the full resume_json in state (it's already on disk)
        c_copy = {k: v for k, v in c.items() if k != "resume_json"}
        c_copy["scores"] = scores
        enriched_candidates.append(c_copy)

    state = {
        "job": {
            "description": job_description,
            "default_vectors_json": default_vectors_json_text,
            "custom_vectors_json": custom_vectors_json_text,
        },
        "candidates": enriched_candidates,
    }

    save_state(state)
    return state
