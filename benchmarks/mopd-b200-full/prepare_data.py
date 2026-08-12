#!/usr/bin/env python3
"""Create the deterministic GSM8K/HumanEval mixture used by every runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


MATH_DATASET = "openai/gsm8k"
MATH_REVISION = "740312add88f781978c0658806c59bc2815b9866"
CODE_DATASET = "openai/openai_humaneval"
CODE_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
MATH_SYSTEM = "You are a careful mathematics solver."
CODE_SYSTEM = "You are an expert Python programmer."


def chat(system: str, prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def math_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "Solve the grade-school math problem. Reason step by step, then give the "
        "final answer as a single number on the last line, prefixed with '#### ' "
        "(e.g. '#### 42').\n\n"
        + row["question"]
    )
    return {
        "domain": "math",
        "source_index": index,
        "prompt": prompt,
        "system": MATH_SYSTEM,
        "answer": row["answer"],
    }


def code_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "Read the following function signature and docstring, and fully implement "
        "the function described. Your response should only contain the code for "
        "this function.\n\n"
        + row["prompt"]
    )
    return {
        "domain": "code",
        "source_index": index,
        "prompt": prompt,
        "system": CODE_SYSTEM,
        "answer": row["canonical_solution"],
        "task_id": row["task_id"],
    }


def interleave(math: list[dict[str, Any]], code: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assert len(math) == len(code)
    return [row for pair in zip(math, code, strict=True) for row in pair]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-domain", type=int, default=128)
    parser.add_argument("--eval-per-domain", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    math_raw = load_dataset(MATH_DATASET, "main", split="train", revision=MATH_REVISION)
    code_raw = load_dataset(CODE_DATASET, split="test", revision=CODE_REVISION)
    required = args.train_per_domain + args.eval_per_domain
    if len(math_raw) < required or len(code_raw) < required:
        raise ValueError("Requested split is larger than one of the source datasets")

    math = [math_row(i, dict(math_raw[i])) for i in range(required)]
    code = [code_row(i, dict(code_raw[i])) for i in range(required)]
    math_train, math_eval = math[: args.train_per_domain], math[args.train_per_domain :]
    code_train, code_eval = code[: args.train_per_domain], code[args.train_per_domain :]
    train = interleave(math_train, code_train)
    eval_rows = interleave(math_eval, code_eval)

    write_jsonl(args.output_dir / "prime_math_train.jsonl", math_train)
    write_jsonl(args.output_dir / "prime_code_train.jsonl", code_train)
    write_jsonl(args.output_dir / "prime_math_eval.jsonl", math_eval)
    write_jsonl(args.output_dir / "prime_code_eval.jsonl", code_eval)

    for split, rows in (("train", train), ("eval", eval_rows)):
        miles = [
            {
                "messages": chat(row["system"], row["prompt"]),
                "label": row["answer"],
                "metadata": {
                    "opd_teacher": row["domain"],
                    "domain": row["domain"],
                    "source_index": row["source_index"],
                },
            }
            for row in rows
        ]
        write_jsonl(args.output_dir / f"miles_{split}.jsonl", miles)

    for split, rows in (("train", train), ("eval", eval_rows)):
        verl = [
            {
                "data_source": row["domain"],
                "prompt": chat(row["system"], row["prompt"]),
                "ability": row["domain"],
                "reward_model": {"style": "rule", "ground_truth": row["answer"]},
                "extra_info": {
                    "domain": row["domain"],
                    "source_index": row["source_index"],
                },
            }
            for row in rows
        ]
        Dataset.from_list(verl).to_parquet(args.output_dir / f"verl_{split}.parquet")

    manifest = {
        "math_dataset": MATH_DATASET,
        "math_revision": MATH_REVISION,
        "math_config": "main",
        "math_split": "train",
        "code_dataset": CODE_DATASET,
        "code_revision": CODE_REVISION,
        "code_split": "test",
        "train_per_domain": args.train_per_domain,
        "eval_per_domain": args.eval_per_domain,
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "ordering": "math/code interleaved",
        "routing": {"math": "math teacher", "code": "code teacher"},
        "system_prompts": {"math": MATH_SYSTEM, "code": CODE_SYSTEM},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
