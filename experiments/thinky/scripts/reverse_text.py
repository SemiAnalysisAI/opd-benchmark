"""Shared reverse-text task definition and scoring."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

DATASET_ID = "PrimeIntellect/Reverse-Text-RL"
SYSTEM_PROMPT = (
    "Reverse the text character-by-character. Put your answer in <reversed_text> tags."
)
REVERSED_TEXT_TAG = re.compile(r"<reversed_text>(.*?)</reversed_text>", re.DOTALL)


def expected_answer(prompt: str) -> str:
    return prompt[::-1]


def score_response(response: str, answer: str) -> tuple[float, bool]:
    """Return the canonical LCS-ratio score and tag-format validity."""
    match = REVERSED_TEXT_TAG.search(response or "")
    prediction = match.group(1).strip() if match else ""
    return SequenceMatcher(None, prediction, answer).ratio(), match is not None


def split_prompts(prompts: list[str], eval_size: int) -> tuple[list[str], list[str]]:
    """Match the deterministic source-order split used by the local benchmark."""
    if eval_size < 0:
        raise ValueError("eval_size must be non-negative")
    if not prompts:
        raise ValueError("the reverse-text dataset is empty")
    if eval_size == 0:
        return prompts, []
    split_at = max(1, len(prompts) - eval_size)
    return prompts[:split_at], prompts[split_at:]
