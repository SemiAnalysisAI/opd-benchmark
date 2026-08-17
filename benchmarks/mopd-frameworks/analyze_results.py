#!/usr/bin/env python3
"""Reduce native Prime-RL, Miles, verl, and optional Slime logs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def parse_env(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        parsed[key] = int(value) if value.isdigit() else value
    return parsed


def gpu_stats(path: Path, start_epoch: int, end_epoch: int) -> dict[str, Any]:
    by_gpu: dict[int, list[dict[str, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(row["epoch"])
            if start_epoch <= epoch <= end_epoch:
                by_gpu[int(row["index"])].append(
                    {
                        "epoch": epoch,
                        "util": float(row["utilization_gpu_pct"]),
                        "memory": float(row["memory_used_mib"]),
                        "power": float(row["power_w"]),
                    }
                )

    result: dict[str, Any] = {}
    total_energy_wh = 0.0
    for index, rows in sorted(by_gpu.items()):
        rows.sort(key=lambda row: row["epoch"])
        energy_wh = 0.0
        for current, following in zip(rows, rows[1:]):
            seconds = min(max(following["epoch"] - current["epoch"], 0), 2)
            energy_wh += current["power"] * seconds / 3600
        if rows:
            energy_wh += rows[-1]["power"] / 3600
        total_energy_wh += energy_wh
        result[str(index)] = {
            "samples": len(rows),
            "avg_utilization_pct": mean([row["util"] for row in rows]),
            "peak_memory_mib": max(row["memory"] for row in rows),
            "avg_power_w": mean([row["power"] for row in rows]),
            "energy_wh": energy_wh,
        }
    return {"gpus": result, "total_energy_wh": total_energy_wh}


def teacher_requests(run_dir: Path, marker: str) -> dict[str, int]:
    counts = {}
    for domain in ("math", "code"):
        path = run_dir / f"{domain}-teacher.log"
        counts[domain] = (
            path.read_text(encoding="utf-8", errors="replace").count(marker)
            if path.exists()
            else 0
        )
    return counts


def common_summary(run_dir: Path) -> dict[str, Any]:
    wall = parse_env(run_dir / "wall_time.env")
    return {
        "wall": wall,
        "physical_teacher_count": 1 if wall.get("teacher_mode") == "single" else 2,
        "gpu": gpu_stats(
            run_dir / "gpu.csv", wall["training_start_epoch"], wall["end_epoch"]
        ),
    }


def metric_triplet(values: list[float]) -> dict[str, float]:
    return {
        "first": values[0],
        "mean": mean(values),
        "final": values[-1],
    }


def parse_prime(run_dir: Path) -> dict[str, Any]:
    summary = common_summary(run_dir)
    trainer: dict[int, dict[str, Any]] = defaultdict(dict)
    with (run_dir / "trainer-metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            trainer[int(row["step"])].update(row)

    orchestrator = []
    with (run_dir / "orchestrator-metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "progress/tokens" in row:
                orchestrator.append(row)
    trainer_rows = [trainer[index] for index in sorted(trainer)]
    orchestrator.sort(key=lambda row: row["step"])

    step_times = [float(row["time/step"]) for row in orchestrator]
    total_tokens = int(sum(row["progress/tokens"] for row in orchestrator))
    output_tokens = int(sum(row["progress/output_tokens"] for row in orchestrator))
    math_trajectories = int(round(sum(row["batch/math"] * 64 for row in orchestrator)))
    code_trajectories = int(round(sum(row["batch/code"] * 64 for row in orchestrator)))
    run_log = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    staleness = [int(value) for value in re.findall(r"Max Off-Policy (\d+)", run_log)]

    reverse_kl = [-float(row["ref_kl/mean"]) for row in trainer_rows]
    loss = [float(row["loss/mean"]) for row in trainer_rows]
    grad_norm = [float(row["optim/grad_norm"]) for row in trainer_rows]
    summary.update(
        {
            "steps": len(orchestrator),
            "trajectories": math_trajectories + code_trajectories,
            "routes": {"math": math_trajectories, "code": code_trajectories},
            "teacher_endpoint_requests": teacher_requests(
                run_dir, "POST /inference/v1/generate"
            ),
            "total_tokens": total_tokens,
            "output_tokens": output_tokens,
            "mean_response_tokens": output_tokens / (math_trajectories + code_trajectories),
            "truncation_ratio": mean(
                [row["train/agg/all/agent/is_truncated/mean"] for row in orchestrator]
            ),
            "reverse_kl_estimate": metric_triplet(reverse_kl),
            "native_loss": metric_triplet(loss),
            "grad_norm": {"mean": mean(grad_norm), "max": max(grad_norm)},
            "active": {
                "seconds": sum(step_times),
                "mean_step_seconds": mean(step_times),
                "median_step_seconds": median(step_times),
                "steady_median_step_seconds": median(step_times[1:]),
                "tokens_per_second": total_tokens / sum(step_times),
                "trajectories_per_second": (math_trajectories + code_trajectories)
                / sum(step_times),
            },
            "trainer": {
                "mean_forward_backward_seconds": mean(
                    [float(row["time/forward_backward"]) for row in trainer_rows]
                ),
                "mean_reported_tokens_per_second": mean(
                    [float(row["perf/throughput"]) for row in trainer_rows]
                ),
            },
            "max_off_policy_updates": max(staleness, default=0),
        }
    )
    return summary


def literal_records(text: str, pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for match in pattern.finditer(ANSI.sub("", text)):
        records[int(match.group(1))] = ast.literal_eval(match.group(2))
    return [records[index] for index in sorted(records)]


def parse_miles(run_dir: Path) -> dict[str, Any]:
    summary = common_summary(run_dir)
    text = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    train = literal_records(
        text, re.compile(r"log_utils\.py:544 - step (\d+): (\{[^\n]+\})")
    )
    perf = literal_records(
        text, re.compile(r"train_metric_utils\.py:50 - perf (\d+): (\{[^\n]+\})")
    )
    rollout = literal_records(
        text, re.compile(r"metrics\.py:89 - perf (\d+): (\{[^\n]+\})")
    )
    if not (len(train) == len(perf) == len(rollout) == 15):
        raise ValueError(
            f"Miles expected 15 records, got train={len(train)} perf={len(perf)} "
            f"rollout={len(rollout)}"
        )

    step_times = [float(row["perf/step_time"]) for row in perf]
    token_counts = [
        float(row["perf/actor_train_tok_per_s"]) * float(row["perf/actor_train_time"])
        for row in perf
    ]
    reverse_kl = [float(row["train/opd_reverse_kl"]) for row in train]
    loss = [float(row["train/loss"]) for row in train]
    grad_norm = [float(row["train/grad_norm"]) for row in train]
    requests = teacher_requests(run_dir, "POST /generate")
    if summary["physical_teacher_count"] == 1:
        routed = {"math": 480, "code": 480}
    else:
        routed = {domain: count - 1 for domain, count in requests.items()}
    trajectories = sum(routed.values())
    response_tokens = sum(row["rollout/response_len/mean"] * 64 for row in rollout)
    summary.update(
        {
            "steps": len(train),
            "trajectories": trajectories,
            "routes": routed,
            "teacher_endpoint_requests": requests,
            "total_tokens": int(round(sum(token_counts))),
            "output_tokens": int(round(response_tokens)),
            "mean_response_tokens": response_tokens / trajectories,
            "truncation_ratio": mean(
                [float(row["rollout/truncated_ratio"]) for row in rollout]
            ),
            "reverse_kl_estimate": metric_triplet(reverse_kl),
            "native_loss": metric_triplet(loss),
            "grad_norm": {"mean": mean(grad_norm), "max": max(grad_norm)},
            "active": {
                "seconds": sum(step_times),
                "mean_step_seconds": mean(step_times),
                "median_step_seconds": median(step_times),
                "steady_median_step_seconds": median(step_times[1:]),
                "tokens_per_second": sum(token_counts) / sum(step_times),
                "trajectories_per_second": trajectories / sum(step_times),
            },
            "trainer": {
                "mean_actor_train_seconds": mean(
                    [float(row["perf/actor_train_time"]) for row in perf]
                ),
                "mean_teacher_and_logprob_seconds": mean(
                    [
                        float(row["perf/ref_log_probs_time"])
                        + float(row["perf/log_probs_time"])
                        for row in perf
                    ]
                ),
            },
            "max_off_policy_updates": 0,
        }
    )
    return summary


def parse_slime(run_dir: Path) -> dict[str, Any]:
    summary = common_summary(run_dir)
    text = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    train = literal_records(
        text, re.compile(r"model\.py:\d+ - step (\d+): (\{[^\n]+\})")
    )
    perf = literal_records(
        text, re.compile(r"train_metric_utils\.py:\d+ - perf (\d+): (\{[^\n]+\})")
    )
    rollout = literal_records(
        text, re.compile(r"rollout\.py:\d+ - perf (\d+): (\{[^\n]+\})")
    )
    expected_steps = int(summary["wall"].get("steps", 15))
    if not (len(train) == len(perf) == len(rollout) == expected_steps):
        raise ValueError(
            f"Slime expected {expected_steps} records, got train={len(train)} "
            f"perf={len(perf)} rollout={len(rollout)}"
        )

    step_times = [float(row["perf/step_time"]) for row in perf]
    token_counts = [
        float(row["perf/actor_train_tok_per_s"]) * float(row["perf/actor_train_time"])
        for row in perf
    ]
    reverse_kl = [
        mean(
            [
                float(row["train/mopd_reverse_kl/math"]),
                float(row["train/mopd_reverse_kl/code"]),
            ]
        )
        for row in train
    ]
    loss = [float(row["train/loss"]) for row in train]
    grad_norm = [float(row["train/grad_norm"]) for row in train]
    requests = teacher_requests(run_dir, "POST /generate")
    routes = {"math": expected_steps * 32, "code": expected_steps * 32}
    trajectories = sum(routes.values())
    response_tokens = sum(float(row["rollout/response_len/mean"]) * 64 for row in rollout)
    summary.update(
        {
            "steps": len(train),
            "trajectories": trajectories,
            "routes": routes,
            "teacher_endpoint_requests": requests,
            "total_tokens": int(round(sum(token_counts))),
            "output_tokens": int(round(response_tokens)),
            "mean_response_tokens": response_tokens / trajectories,
            "truncation_ratio": mean(
                [float(row["rollout/truncated_ratio"]) for row in rollout]
            ),
            "reverse_kl_estimate": metric_triplet(reverse_kl),
            "native_loss": metric_triplet(loss),
            "grad_norm": {"mean": mean(grad_norm), "max": max(grad_norm)},
            "active": {
                "seconds": sum(step_times),
                "mean_step_seconds": mean(step_times),
                "median_step_seconds": median(step_times),
                "steady_median_step_seconds": median(step_times[1:]),
                "tokens_per_second": sum(token_counts) / sum(step_times),
                "trajectories_per_second": trajectories / sum(step_times),
            },
            "trainer": {
                "mean_actor_train_seconds": mean(
                    [float(row["perf/actor_train_time"]) for row in perf]
                ),
                "mean_teacher_and_logprob_seconds": mean(
                    [
                        float(row.get("perf/ref_log_probs_time", 0))
                        + float(row.get("perf/log_probs_time", 0))
                        for row in perf
                    ]
                ),
                "mean_importance_weight": mean(
                    [float(row["train/mopd_is_weight_mean"]) for row in train]
                ),
                "mean_importance_nonzero_fraction": mean(
                    [float(row["train/mopd_is_nonzero_frac"]) for row in train]
                ),
            },
            "max_off_policy_updates": 0,
        }
    )
    return summary


def parse_verl_line(line: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for part in line.split(" - "):
        if ":" not in part:
            continue
        key, value = part.rsplit(":", 1)
        try:
            parsed[key.rsplit(") ", 1)[-1]] = float(value)
        except ValueError:
            continue
    return parsed


def parse_verl(run_dir: Path) -> dict[str, Any]:
    summary = common_summary(run_dir)
    rows = []
    for line in ANSI.sub(
        "", (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    ).splitlines():
        if re.search(r"\) step:\d+ - global_seqlen/min:", line):
            rows.append(parse_verl_line(line))
    if len(rows) != 15:
        raise ValueError(f"verl expected 15 records, got {len(rows)}")

    step_times = [row["timing_s/step"] for row in rows]
    token_counts = [row["perf/total_num_tokens"] for row in rows]
    response_tokens = sum(row["response_length/mean"] * 64 for row in rows)
    reverse_kl = [row["actor/distillation/loss"] for row in rows]
    loss = [row["actor/loss"] for row in rows]
    grad_norm = [row["actor/grad_norm"] for row in rows]
    requests = teacher_requests(run_dir, "POST /generate")
    if summary["physical_teacher_count"] == 1:
        routed = {"math": 480, "code": 480}
    else:
        routed = {domain: count - 1 for domain, count in requests.items()}
    trajectories = sum(routed.values())
    summary.update(
        {
            "steps": len(rows),
            "trajectories": trajectories,
            "routes": routed,
            "teacher_endpoint_requests": requests,
            "total_tokens": int(sum(token_counts)),
            "output_tokens": int(round(response_tokens)),
            "mean_response_tokens": response_tokens / trajectories,
            "truncation_ratio": mean([row["response_length/clip_ratio"] for row in rows]),
            "reverse_kl_estimate": metric_triplet(reverse_kl),
            "native_loss": metric_triplet(loss),
            "grad_norm": {"mean": mean(grad_norm), "max": max(grad_norm)},
            "active": {
                "seconds": sum(step_times),
                "mean_step_seconds": mean(step_times),
                "median_step_seconds": median(step_times),
                "steady_median_step_seconds": median(step_times[1:]),
                "tokens_per_second": sum(token_counts) / sum(step_times),
                "trajectories_per_second": trajectories / sum(step_times),
            },
            "trainer": {
                "mean_generation_seconds": mean([row["timing_s/gen"] for row in rows]),
                "mean_actor_update_seconds": mean(
                    [row["timing_s/update_actor"] for row in rows]
                ),
                "mean_weight_update_seconds": mean(
                    [row["timing_s/update_weights"] for row in rows]
                ),
            },
            "max_off_policy_updates": int(
                max(row["training/off_policy/trajectory_staleness_worst/max"] for row in rows)
            ),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).parent / "results"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frameworks = {
        "prime-rl": parse_prime(args.results_dir / "prime"),
        "miles": parse_miles(args.results_dir / "miles"),
        "verl": parse_verl(args.results_dir / "verl"),
    }
    if (args.results_dir / "slime" / "wall_time.env").exists():
        frameworks["slime"] = parse_slime(args.results_dir / "slime")

    result = {
        "recipe": {
            "steps": 15,
            "prompts_per_step": 16,
            "rollouts_per_prompt": 4,
            "trajectories_per_step": 64,
            "total_trajectories": 960,
        },
        "frameworks": frameworks,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
