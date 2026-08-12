#!/usr/bin/env python3
"""Validate data equivalence, routing, and tokenizer-ID compatibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


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

    names = ("student", "math-teacher", "code-teacher")
    tokenizers = {
        name: AutoTokenizer.from_pretrained(args.model_dir / name, use_fast=True)
        for name in names
    }
    reference = tokenizers["student"]
    samples = [
        "Hello, world!",
        "The answer is 42.",
        "def fibonacci(n: int) -> int:\n    pass",
        "数学とcode: αβγ 🚀",
    ] + [row["prompt"] for row in prime_math[:8] + prime_code[:8]]
    semantic = {}
    for name, tokenizer in tokenizers.items():
        assert len(tokenizer) == len(reference)
        assert tokenizer.vocab_size == reference.vocab_size
        assert tokenizer.get_vocab() == reference.get_vocab()
        assert tokenizer.get_added_vocab() == reference.get_added_vocab()
        assert tokenizer.all_special_ids == reference.all_special_ids
        assert tokenizer.eos_token_id == reference.eos_token_id
        assert tokenizer.pad_token_id == reference.pad_token_id
        assert [tokenizer.encode(sample, add_special_tokens=False) for sample in samples] == [
            reference.encode(sample, add_special_tokens=False) for sample in samples
        ]
        semantic[name] = {
            "length": len(tokenizer),
            "vocab_size": tokenizer.vocab_size,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "tokenizer_json_sha256": digest(args.model_dir / name / "tokenizer.json"),
        }

    print(
        json.dumps(
            {
                "status": "ok",
                "rows": len(miles),
                "routes": Counter(row["metadata"]["opd_teacher"] for row in miles),
                "tokenizer_contract": "semantic token-to-ID equivalence",
                "tokenizers": semantic,
            },
            indent=2,
            default=dict,
        )
    )


if __name__ == "__main__":
    main()
