"""Canonical reverse-text reward used by the Prime-RL task environment."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


_TAG = re.compile(r"<reversed_text>(.*?)</reversed_text>", re.DOTALL)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **_: object,
) -> dict[str, float]:
    """Return the LCS-ratio score used by reverse_text_v1 in Prime-RL."""
    del data_source, extra_info
    match = _TAG.search(solution_str or "")
    response = match.group(1).strip() if match else ""
    score = SequenceMatcher(None, response, ground_truth).ratio()
    return {"score": score, "acc": score}
