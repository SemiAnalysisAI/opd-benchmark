#!/usr/bin/env python3
"""Fail fast when the pinned Slime source, models, or MOPD data drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


SLIME_COMMIT = "f51403558a47290d190fc9dfabe1859be73aca4f"
EXPECTED_ROWS = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--slime-dir", type=Path, required=True)
    args = parser.parse_args()

    actual_commit = subprocess.check_output(
        ["git", "-C", str(args.slime_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != SLIME_COMMIT:
        raise ValueError(f"Slime commit drift: expected {SLIME_COMMIT}, got {actual_commit}")

    required_source = {
        "slime/rollout/mopd.py": ("mopd_domains", "MOPD_TEACHER_URLS"),
        "slime/utils/arguments.py": ("--use-mopd", "--mopd-distill-type"),
    }
    for relative_path, markers in required_source.items():
        content = (args.slime_dir / relative_path).read_text(encoding="utf-8")
        for marker in markers:
            if marker not in content:
                raise ValueError(f"{relative_path} is missing {marker!r}")

    data_path = args.data_dir / "slime_train.jsonl"
    counts: Counter[str] = Counter()
    with data_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} training rows, got {len(rows)}")
    for index, row in enumerate(rows):
        metadata = row["metadata"]
        domain = metadata["domain"]
        expected_domain = "math" if index % 2 == 0 else "code"
        if domain != expected_domain:
            raise ValueError(f"row {index}: expected {expected_domain}, got {domain}")
        if metadata.get("mopd_domains") != [domain]:
            raise ValueError(f"row {index}: invalid mopd_domains routing")
        counts[domain] += 1
    if counts != Counter({"math": 128, "code": 128}):
        raise ValueError(f"unexpected domain counts: {dict(counts)}")

    tokenizers = {
        name: sha256(args.model_dir / name / "tokenizer.json")
        for name in ("student", "math-teacher", "code-teacher")
    }
    if len(set(tokenizers.values())) != 1:
        raise ValueError(f"tokenizer mismatch: {tokenizers}")

    print(
        json.dumps(
            {
                "slime_commit": actual_commit,
                "rows": len(rows),
                "domains": counts,
                "tokenizer_sha256": next(iter(tokenizers.values())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
