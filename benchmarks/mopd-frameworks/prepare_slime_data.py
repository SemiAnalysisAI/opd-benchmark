#!/usr/bin/env python3
"""Adapt the shared MOPD JSONL data to Slime's per-sample routing format."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DOMAINS = {"math", "code"}


def adapt_row(row: dict[str, Any], line_number: int) -> dict[str, Any]:
    messages = row.get("messages")
    metadata = row.get("metadata")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"line {line_number}: messages must be a non-empty list")
    if not isinstance(metadata, dict):
        raise ValueError(f"line {line_number}: metadata must be an object")

    domain = metadata.get("domain")
    if domain not in DOMAINS:
        raise ValueError(f"line {line_number}: unsupported domain {domain!r}")
    if metadata.get("opd_teacher") != domain:
        raise ValueError(f"line {line_number}: opd_teacher does not match domain")

    adapted = dict(row)
    adapted_metadata = dict(metadata)
    adapted_metadata["mopd_domains"] = [domain]
    adapted["metadata"] = adapted_metadata
    return adapted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = adapt_row(json.loads(line), line_number)
            rows.append(row)
            counts[row["metadata"]["domain"]] += 1

    if not rows:
        raise ValueError("input contains no rows")
    if set(counts) != DOMAINS or len(set(counts.values())) != 1:
        raise ValueError(f"expected an equal math/code mixture, got {dict(counts)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "rows": len(rows),
                "domains": counts,
            }
        )
    )


if __name__ == "__main__":
    main()
