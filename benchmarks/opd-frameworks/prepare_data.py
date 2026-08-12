#!/usr/bin/env python3
"""Create equivalent reverse-text inputs for verl and Miles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset


DATASET = "PrimeIntellect/Reverse-Text-RL"
SYSTEM = "Reverse the text character-by-character. Put your answer in <reversed_text> tags."


def messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-size", type=int, default=128)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(DATASET, split="train")
    indexed = [{"index": i, "prompt": row["prompt"]} for i, row in enumerate(raw)]

    # Preserve the source order for training. Reserve the tail for a deterministic
    # common evaluation set without changing Prime-RL's source task semantics.
    split_at = max(1, len(indexed) - args.eval_size)
    train_rows = indexed[:split_at]
    eval_rows = indexed[split_at:]

    verl_train = []
    verl_eval = []
    miles_train = []
    miles_eval = []
    common_eval = []

    for split_name, source_rows, verl_out, miles_out in (
        ("train", train_rows, verl_train, miles_train),
        ("eval", eval_rows, verl_eval, miles_eval),
    ):
        for row in source_rows:
            prompt = row["prompt"]
            answer = prompt[::-1]
            chat = messages(prompt)
            metadata = {"source_index": row["index"], "split": split_name}
            verl_out.append(
                {
                    "data_source": "reverse-text",
                    "prompt": chat,
                    "ability": "reverse-text",
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": metadata,
                }
            )
            miles_out.append({"messages": chat, "label": answer, "metadata": metadata})
            if split_name == "eval":
                common_eval.append(
                    {"messages": chat, "prompt": prompt, "answer": answer, "metadata": metadata}
                )

    Dataset.from_list(verl_train).to_parquet(args.output_dir / "train.parquet")
    Dataset.from_list(verl_eval).to_parquet(args.output_dir / "eval.parquet")

    for path, rows in (
        (args.output_dir / "train.jsonl", miles_train),
        (args.output_dir / "eval.jsonl", miles_eval),
        (args.output_dir / "common_eval.jsonl", common_eval),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": DATASET,
        "source_rows": len(indexed),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "system_prompt": SYSTEM,
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

