#!/usr/bin/env python3
"""Fail if the three framework inputs or tokenizer contract diverge."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    miles = jsonl(args.data_dir / "miles_train.jsonl")
    verl = list(Dataset.from_parquet(str(args.data_dir / "verl_train.parquet")))
    assert len(miles) == len(verl) == 256
    for index, (miles_row, verl_row) in enumerate(zip(miles, verl, strict=True)):
        expected = "math" if index % 2 == 0 else "code"
        assert miles_row["metadata"]["opd_teacher"] == expected
        assert verl_row["data_source"] == expected
        assert miles_row["messages"] == verl_row["prompt"]
        assert miles_row["metadata"]["source_index"] == verl_row["extra_info"]["source_index"]

    prime_math = jsonl(args.data_dir / "prime_math_train.jsonl")
    prime_code = jsonl(args.data_dir / "prime_code_train.jsonl")
    assert len(prime_math) == len(prime_code) == 128
    assert all(row["domain"] == "math" for row in prime_math)
    assert all(row["domain"] == "code" for row in prime_code)
    for domain_rows, expected_rows in ((prime_math, miles[0::2]), (prime_code, miles[1::2])):
        for prime_row, mixed_row in zip(domain_rows, expected_rows, strict=True):
            assert mixed_row["messages"] == [
                {"role": "system", "content": prime_row["system"]},
                {"role": "user", "content": prime_row["prompt"]},
            ]

    hashes = {
        name: digest(args.model_dir / name / "tokenizer.json")
        for name in ("student", "math-teacher", "code-teacher")
    }
    assert len(set(hashes.values())) == 1, hashes
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": len(miles),
                "routes": Counter(row["metadata"]["opd_teacher"] for row in miles),
                "tokenizer_sha256": next(iter(hashes.values())),
            },
            indent=2,
            default=dict,
        )
    )


if __name__ == "__main__":
    main()

