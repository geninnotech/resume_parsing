from __future__ import annotations

import json
import os
from typing import Dict, Any, Tuple, List

from config import PROMPTS_DIR, DATA_DIR
from llm.llm_modules import openaiLLM
from services.state import load_state


def _read_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def generate_vector_weights_for_query(
    user_query: str,
    openai_fast: openaiLLM,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Use GPT-5-mini to produce weights per vector for the given query.
    Returns (weights_dict, state)
    """
    state = load_state()
    if state is None:
        raise RuntimeError("No state found. Upload resumes & JD first.")

    job = state["job"]
    job_description = job["description"]
    default_vectors_json_text = job["default_vectors_json"]
    custom_vectors_json_text = job["custom_vectors_json"]

    template = _read_prompt("generate_vector_weights.txt")

    prompt = (
        template.replace("<#job_description#>", job_description)
        .replace("<#default_weights#>", default_vectors_json_text)
        .replace("<#custom_vectors#>", custom_vectors_json_text)
        .replace("<#user_query#>", user_query)
    )

    raw = openai_fast.infer(
        prompt=prompt,
        reasoning_effort="minimal",
    )

    weights = json.loads(raw)
    # weights: { "vector_name": int }

    # Normalize to ints
    weights = {k: int(v) for k, v in weights.items()}
    return weights, state


def compute_query_scores(
    weights: Dict[str, int],
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Combine per-vector candidate scores with query weights.
    Returns candidates sorted by query_score desc, each with query_score.
    """
    ranked: List[Dict[str, Any]] = []

    for c in state["candidates"]:
        scores = c.get("scores", {})
        default_scores = scores.get("default_vectors", {})
        custom_scores = scores.get("custom_vectors", {})

        total = 0
        # default + custom
        for vec_name, score in {**default_scores, **custom_scores}.items():
            w = weights.get(vec_name, 0)
            total += w * score  # both 0–100

        c_with_score = dict(c)
        c_with_score["query_score"] = total
        ranked.append(c_with_score)

    ranked.sort(key=lambda x: x["query_score"], reverse=True)
    return ranked


def build_answer_for_query(
    user_query: str,
    ranked_candidates: List[Dict[str, Any]],
    top_k: int,
    openai_fast: openaiLLM,
) -> str:
    """
    Use GPT-5-mini to generate a natural language answer that refers to
    candidates only via tags like <#resume-2#>.
    """
    top = ranked_candidates[:top_k]

    blocks = []
    for rank, c in enumerate(top, start=1):
        cid = c["id"]
        name = c["name"]
        score = c["query_score"]

        # Load markdown
        md_path = DATA_DIR / c["markdown_path"]
        with md_path.open("r", encoding="utf-8") as f:
            md_text = f.read()

        block = [
            f"Candidate rank: {rank}",
            f"ID tag: <#resume-{cid}#>",
            f"Display_name_hint: {name}",
            f"Query_score: {score}",
            "",
            "Resume_markdown:",
            "```markdown",
            md_text,
            "```",
            "",
        ]
        blocks.append("\n".join(block))

    candidates_blob = "\n\n".join(blocks)

    prompt = f"""
You are a hiring assistant. A user has asked a query about what kind of candidate they want.

User query:
\"\"\"{user_query}\"\"\"

You are given the top {top_k} candidates that may match this query. Each candidate has:
- An ID tag like <#resume-2#> (this is how the UI identifies that candidate).
- A display_name_hint (for your understanding only; do NOT output it directly).
- A query_score.
- Their full resume in Markdown.

YOUR RULES:
- When you talk about a candidate, ALWAYS refer to them ONLY by their ID tag, e.g. <#resume-2#>.
- Do NOT print their human name; the UI will replace tags with clickable names.
- Provide a concise comparison: who fits best, why, and any tradeoffs.
- Prefer bullet points and short paragraphs.
- If multiple candidates are suitable, rank them and explain briefly.

CANDIDATES:
{candidates_blob}
"""

    answer = openai_fast.infer(
        prompt=prompt,
        reasoning_effort="minimal",
    )
    return answer
