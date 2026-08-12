"""Post-hoc reverse-text scoring for Miles rollouts.

Miles' external-teacher OPD reward function intentionally returns zero task
reward. This hook observes completed samples and logs the canonical task score
without changing the rewards or advantages consumed by training.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Iterable


_TAG = re.compile(r"<reversed_text>(.*?)</reversed_text>", re.DOTALL)


def _flatten(items: Iterable[Any]) -> Iterable[Any]:
    for item in items:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


def process_samples(args: Any, all_samples: list[Any], data_source: Any) -> None:
    """Log task quality for completed samples without mutating them."""
    del args, data_source
    scores: list[float] = []
    truncated = 0
    for sample in _flatten(all_samples):
        label = getattr(sample, "label", None)
        if label is None:
            continue
        response = getattr(sample, "response", "") or ""
        match = _TAG.search(response)
        prediction = match.group(1).strip() if match else ""
        scores.append(SequenceMatcher(None, prediction, label).ratio())
        status = getattr(sample, "status", None)
        truncated += int(getattr(status, "value", status) == "truncated")

    mean = sum(scores) / len(scores) if scores else 0.0
    truncation_rate = truncated / len(scores) if scores else 0.0
    print(
        "OPD_BENCH_TASK_SCORE "
        f"count={len(scores)} mean={mean:.12f} "
        f"truncation_rate={truncation_rate:.12f}",
        flush=True,
    )
