#!/usr/bin/env python3
"""Cross-score fixed student rollouts with both MOPD specialist policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def post_json(url: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} returned HTTP {error.code}: {body}") from error


def token_logprobs(meta: dict[str, Any], field: str) -> tuple[list[float], list[int]]:
    entries = meta.get(field)
    if not entries:
        raise ValueError(f"SGLang response is missing meta_info.{field}")
    values: list[float] = []
    token_ids: list[int] = []
    for entry in entries:
        if entry is None or entry[0] is None:
            continue
        values.append(float(entry[0]))
        token_ids.append(int(entry[1]))
    return values, token_ids


def generate(
    url: str,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> tuple[list[int], list[float], str, str]:
    response = post_json(
        url,
        {
            "input_ids": prompt_ids,
            "sampling_params": {
                "temperature": temperature,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
                "skip_special_tokens": False,
                "sampling_seed": seed,
            },
            "return_logprob": True,
        },
    )
    logprobs, output_ids = token_logprobs(
        response["meta_info"], "output_token_logprobs"
    )
    if len(logprobs) != len(output_ids):
        raise ValueError("Student output token/log-prob length mismatch")
    finish = response["meta_info"].get("finish_reason")
    if isinstance(finish, dict):
        finish = str(finish.get("type", finish))
    return output_ids, logprobs, str(response.get("text", "")), str(finish)


def score(url: str, prompt_ids: list[int], output_ids: list[int]) -> list[float]:
    response = post_json(
        url,
        {
            "input_ids": prompt_ids + output_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            # Include one prompt token so every response token receives a causal
            # next-token log probability, including the first response token.
            "logprob_start_len": max(len(prompt_ids) - 1, 0),
        },
    )
    values, scored_ids = token_logprobs(
        response["meta_info"], "input_token_logprobs"
    )
    values = values[-len(output_ids) :]
    scored_ids = scored_ids[-len(output_ids) :]
    if scored_ids != output_ids:
        raise ValueError(
            "Teacher token alignment mismatch: "
            f"expected {output_ids[:8]}... got {scored_ids[:8]}..."
        )
    return values


def load_rows(path: Path, prompts_per_domain: int) -> list[dict[str, Any]]:
    selected: dict[str, list[dict[str, Any]]] = {"math": [], "code": []}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            domain = str(row["metadata"]["domain"])
            if domain in selected and len(selected[domain]) < prompts_per_domain:
                selected[domain].append(row)
            if all(len(rows) == prompts_per_domain for rows in selected.values()):
                break
    if any(len(rows) != prompts_per_domain for rows in selected.values()):
        raise ValueError(f"Could not load {prompts_per_domain} prompts per domain")
    return [
        row
        for index in range(prompts_per_domain)
        for row in (selected["math"][index], selected["code"][index])
    ]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_prompt_ci(
    records: list[dict[str, Any]], *, iterations: int = 10_000
) -> list[float]:
    by_prompt: dict[int, list[float]] = {}
    for row in records:
        by_prompt.setdefault(int(row["source_index"]), []).append(
            float(row["specialist_logprob_advantage"])
        )
    prompt_means = [statistics.fmean(values) for values in by_prompt.values()]
    rng = random.Random(42)
    boot = [
        statistics.fmean(rng.choices(prompt_means, k=len(prompt_means)))
        for _ in range(iterations)
    ]
    return [percentile(boot, 0.025), percentile(boot, 0.975)]


def summarize_domain(records: list[dict[str, Any]]) -> dict[str, Any]:
    token_count = sum(int(row["response_tokens"]) for row in records)
    specialist_total = sum(float(row["specialist_logprob_sum"]) for row in records)
    mismatched_total = sum(float(row["mismatched_logprob_sum"]) for row in records)
    advantages = [float(row["specialist_logprob_advantage"]) for row in records]
    return {
        "trajectories": len(records),
        "prompts": len({int(row["source_index"]) for row in records}),
        "response_tokens": token_count,
        "specialist_mean_token_logprob": specialist_total / token_count,
        "mismatched_mean_token_logprob": mismatched_total / token_count,
        "token_weighted_specialist_advantage": (specialist_total - mismatched_total)
        / token_count,
        "mean_trajectory_specialist_advantage": statistics.fmean(advantages),
        "prompt_cluster_bootstrap_95pct_ci": bootstrap_prompt_ci(records),
        "specialist_win_rate": sum(value > 0 for value in advantages) / len(advantages),
        "mean_sampled_reverse_kl_to_specialist": statistics.fmean(
            float(row["sampled_reverse_kl_to_specialist"]) for row in records
        ),
        "mean_sampled_reverse_kl_to_mismatched": statistics.fmean(
            float(row["sampled_reverse_kl_to_mismatched"]) for row in records
        ),
        "truncation_rate": sum(row["finish_reason"] == "length" for row in records)
        / len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--student-url", default="http://127.0.0.1:13140/generate")
    parser.add_argument("--math-url", default="http://127.0.0.1:13141/generate")
    parser.add_argument("--code-url", default="http://127.0.0.1:13142/generate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts-per-domain", type=int, default=8)
    parser.add_argument("--rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    rows = load_rows(args.data, args.prompts_per_domain)
    records: list[dict[str, Any]] = []
    started = time.time()

    for prompt_position, row in enumerate(rows):
        messages = row["messages"]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(prompt_ids, "keys") and "input_ids" in prompt_ids:
            prompt_ids = prompt_ids["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        domain = str(row["metadata"]["domain"])
        source_index = int(row["metadata"]["source_index"])
        for rollout_index in range(args.rollouts_per_prompt):
            seed = 42_000 + prompt_position * 100 + rollout_index
            output_ids, student_logprobs, output_text, finish_reason = generate(
                args.student_url,
                list(prompt_ids),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=seed,
            )
            if not output_ids:
                raise ValueError(f"Empty rollout for {domain} source {source_index}")
            with ThreadPoolExecutor(max_workers=2) as executor:
                math_future = executor.submit(
                    score, args.math_url, list(prompt_ids), output_ids
                )
                code_future = executor.submit(
                    score, args.code_url, list(prompt_ids), output_ids
                )
                math_logprobs = math_future.result()
                code_logprobs = code_future.result()

            student_mean = statistics.fmean(student_logprobs)
            math_sum = sum(math_logprobs)
            code_sum = sum(code_logprobs)
            specialist = math_logprobs if domain == "math" else code_logprobs
            mismatched = code_logprobs if domain == "math" else math_logprobs
            specialist_sum = sum(specialist)
            mismatched_sum = sum(mismatched)
            record = {
                "domain": domain,
                "source_index": source_index,
                "rollout_index": rollout_index,
                "seed": seed,
                "prompt_tokens": len(prompt_ids),
                "response_tokens": len(output_ids),
                "finish_reason": finish_reason,
                "student_mean_token_logprob": student_mean,
                "math_teacher_mean_token_logprob": math_sum / len(output_ids),
                "code_teacher_mean_token_logprob": code_sum / len(output_ids),
                "specialist_logprob_sum": specialist_sum,
                "mismatched_logprob_sum": mismatched_sum,
                "specialist_logprob_advantage": (specialist_sum - mismatched_sum)
                / len(output_ids),
                "sampled_reverse_kl_to_specialist": student_mean
                - specialist_sum / len(output_ids),
                "sampled_reverse_kl_to_mismatched": student_mean
                - mismatched_sum / len(output_ids),
                "output_token_ids": output_ids,
                "output_sha256": hashlib.sha256(
                    bytes().join(int(token).to_bytes(4, "little") for token in output_ids)
                ).hexdigest(),
                "output_preview": output_text[:240],
            }
            records.append(record)
            print(
                f"{len(records):03d}/{len(rows) * args.rollouts_per_prompt} "
                f"{domain}:{source_index} r{rollout_index} tokens={len(output_ids)} "
                f"specialist_advantage={record['specialist_logprob_advantage']:+.4f}",
                flush=True,
            )

    by_domain = {
        domain: summarize_domain([row for row in records if row["domain"] == domain])
        for domain in ("math", "code")
    }
    total_tokens = sum(int(row["response_tokens"]) for row in records)
    total_advantage = sum(
        float(row["specialist_logprob_sum"])
        - float(row["mismatched_logprob_sum"])
        for row in records
    )
    summary = {
        "recipe": {
            "student_model": args.student_model,
            "data": str(args.data),
            "prompts_per_domain": args.prompts_per_domain,
            "rollouts_per_prompt": args.rollouts_per_prompt,
            "trajectories": len(records),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": 1.0,
            "seed_scheme": "42000 + interleaved_prompt_position*100 + rollout_index",
        },
        "domains": by_domain,
        "overall": {
            "trajectories": len(records),
            "response_tokens": total_tokens,
            "token_weighted_specialist_advantage": total_advantage / total_tokens,
            "specialist_win_rate": sum(
                float(row["specialist_logprob_advantage"]) > 0 for row in records
            )
            / len(records),
            "elapsed_seconds": time.time() - started,
        },
        "interpretation": (
            "Positive specialist advantage means the domain-routed policy assigned "
            "higher average log probability to the same student response than the "
            "other policy. This is a routing-signal probe, not an accuracy metric."
        ),
    }
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
