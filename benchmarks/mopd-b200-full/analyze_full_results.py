#!/usr/bin/env python3
"""Reduce the native logs from the 8xB200 full-size MOPD benchmark."""

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
EXPECTED_STEPS = 15
TRAJECTORIES_PER_STEP = 64


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def triplet(values: list[float]) -> dict[str, float]:
    return {"first": values[0], "mean": mean(values), "final": values[-1]}


def parse_env(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        try:
            result[key] = int(value)
        except ValueError:
            result[key] = value
    return result


def teacher_requests(run_dir: Path, marker: str) -> dict[str, int]:
    result = {}
    for domain in ("math", "code"):
        path = run_dir / f"{domain}-teacher.log"
        result[domain] = path.read_text(encoding="utf-8", errors="replace").count(marker)
    return result


def read_gpu(path: Path, start: int, end: int) -> dict[int, list[dict[str, float]]]:
    by_gpu: dict[int, list[dict[str, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(row["epoch"])
            if start <= epoch <= end:
                by_gpu[int(row["index"])].append(
                    {
                        "epoch": epoch,
                        "util": float(row["utilization_gpu_pct"]),
                        "memory": float(row["memory_used_mib"]),
                        "power": float(row["power_w"]),
                    }
                )
    return by_gpu


def gpu_summary(
    path: Path, start: int, end: int, roles: dict[str, list[int]]
) -> dict[str, Any]:
    by_gpu = read_gpu(path, start, end)
    devices: dict[str, Any] = {}
    total_energy = 0.0
    for index, rows in sorted(by_gpu.items()):
        rows.sort(key=lambda row: row["epoch"])
        energy = 0.0
        for current, following in zip(rows, rows[1:]):
            seconds = min(max(following["epoch"] - current["epoch"], 0), 2)
            energy += current["power"] * seconds / 3600
        if rows:
            energy += rows[-1]["power"] / 3600
        total_energy += energy
        devices[str(index)] = {
            "samples": len(rows),
            "avg_utilization_pct": mean([row["util"] for row in rows]),
            "peak_memory_mib": max(row["memory"] for row in rows),
            "avg_power_w": mean([row["power"] for row in rows]),
            "energy_wh": energy,
        }

    role_stats = {}
    for role, indices in roles.items():
        rows = [row for index in indices for row in by_gpu.get(index, [])]
        role_stats[role] = {
            "gpu_count": len(indices),
            "avg_utilization_pct": mean([row["util"] for row in rows]),
            "peak_memory_mib_per_gpu": max((row["memory"] for row in rows), default=0.0),
            "avg_power_w_per_gpu": mean([row["power"] for row in rows]),
        }
    return {"devices": devices, "roles": role_stats, "total_energy_wh": total_energy}


def base(run_dir: Path, roles: dict[str, list[int]]) -> dict[str, Any]:
    wall = parse_env(run_dir / "wall_time.env")
    return {
        "wall": wall,
        "gpu": gpu_summary(
            run_dir / "gpu.csv",
            wall["training_start_epoch"],
            wall["end_epoch"],
            roles,
        ),
    }


def check_steps(name: str, rows: list[Any]) -> None:
    if len(rows) != EXPECTED_STEPS:
        raise ValueError(f"{name}: expected {EXPECTED_STEPS} steps, got {len(rows)}")


def parse_prime(run_dir: Path) -> dict[str, Any]:
    result = base(run_dir, {"trainer": [0, 1], "rollout": [2, 3, 4, 5], "teachers": [6, 7]})
    trainer_by_step: dict[int, dict[str, Any]] = defaultdict(dict)
    trainer_path = run_dir / "trainer-metrics.jsonl"
    if not trainer_path.exists():
        trainer_path = run_dir / "output" / "metrics.jsonl"
    with trainer_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            trainer_by_step[int(row["step"])].update(row)
    trainer = [trainer_by_step[step] for step in sorted(trainer_by_step)]

    orchestrator = []
    orchestrator_path = run_dir / "orchestrator-metrics.jsonl"
    if not orchestrator_path.exists():
        orchestrator_path = run_dir / "output" / "run_default" / "metrics.jsonl"
    with orchestrator_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "progress/tokens" in row:
                orchestrator.append(row)
    orchestrator.sort(key=lambda row: row["step"])
    check_steps("Prime trainer", trainer)
    check_steps("Prime orchestrator", orchestrator)

    trajectories = sum(int(row["progress/rollouts"]) for row in orchestrator)
    total_tokens = sum(int(row["progress/tokens"]) for row in orchestrator)
    output_tokens = sum(int(row["progress/output_tokens"]) for row in orchestrator)
    pipeline_times = [float(row["time/step"]) for row in orchestrator]
    reverse_kl = [-float(row["ref_kl/mean"]) for row in trainer]
    run_log = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    stale = [int(value) for value in re.findall(r"Max Off-Policy (\d+)", run_log)]
    requests = teacher_requests(run_dir, "POST /inference/v1/generate")
    result.update(
        {
            "steps": len(orchestrator),
            "trajectories": trajectories,
            "routes": {
                "math": int(round(sum(row["batch/math"] * row["progress/rollouts"] for row in orchestrator))),
                "code": int(round(sum(row["batch/code"] * row["progress/rollouts"] for row in orchestrator))),
            },
            "teacher_endpoint_requests": requests,
            "teacher_scored_trajectories": sum(requests.values()),
            "teacher_scoring_overrun": sum(requests.values()) - trajectories,
            "total_tokens": total_tokens,
            "output_tokens": output_tokens,
            "mean_response_tokens": output_tokens / trajectories,
            "truncation_ratio": mean(
                [float(row["train/agg/all/agent/is_truncated/mean"]) for row in orchestrator]
            ),
            "reverse_kl_estimate": triplet(reverse_kl),
            "native_loss": triplet([float(row["loss/mean"]) for row in trainer]),
            "grad_norm": {
                "mean": mean([float(row["optim/grad_norm"]) for row in trainer]),
                "max": max(float(row["optim/grad_norm"]) for row in trainer),
            },
            "pipeline": {
                "seconds": sum(pipeline_times),
                "mean_step_seconds": mean(pipeline_times),
                "median_step_seconds": median(pipeline_times),
                "steady_median_step_seconds": median(pipeline_times[1:]),
                "trajectories_per_second": trajectories / sum(pipeline_times),
                "tokens_per_second": total_tokens / sum(pipeline_times),
            },
            "components": {
                "mean_trainer_step_seconds": mean([float(row["time/step"]) for row in trainer]),
                "mean_wait_for_batch_seconds": mean([float(row["time/wait_for_batch"]) for row in trainer]),
                "mean_forward_backward_seconds": mean([float(row["time/forward_backward"]) for row in trainer]),
                "mean_weight_broadcast_seconds": mean([float(row["time/broadcast_weights"]) for row in trainer]),
                "mean_trainer_tokens_per_second": mean([float(row["perf/throughput"]) for row in trainer]),
                "mean_trainer_mfu_pct": mean([float(row["perf/mfu"]) for row in trainer]),
            },
            "max_off_policy_updates": max(stale, default=0),
            "observability_note": "Prime's Trainable counter is reward-advantage-only; OPD rows still train.",
        }
    )
    return result


def literal_records(text: str, pattern: str) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for match in re.finditer(pattern, ANSI.sub("", text)):
        rows[int(match.group(1))] = ast.literal_eval(match.group(2))
    return [rows[index] for index in sorted(rows)]


def parse_miles(run_dir: Path) -> dict[str, Any]:
    result = base(run_dir, {"trainer": [0, 1], "rollout": [2, 3, 4, 5], "teachers": [6, 7]})
    text = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    train = literal_records(text, r"log_utils\.py:544 - step (\d+): (\{[^\n]+\})")
    perf = literal_records(text, r"train_metric_utils\.py:50 - perf (\d+): (\{[^\n]+\})")
    rollout = literal_records(text, r"metrics\.py:89 - perf (\d+): (\{[^\n]+\})")
    check_steps("Miles train", train)
    check_steps("Miles perf", perf)
    check_steps("Miles rollout", rollout)
    pipeline_times = [float(row["perf/step_time"]) for row in perf]
    token_counts = [
        float(row["perf/actor_train_tok_per_s"]) * float(row["perf/actor_train_time"])
        for row in perf
    ]
    requests = teacher_requests(run_dir, "POST /generate")
    routes = {domain: count - 1 for domain, count in requests.items()}
    trajectories = sum(routes.values())
    output_tokens = sum(float(row["rollout/response_len/mean"]) * TRAJECTORIES_PER_STEP for row in rollout)
    result.update(
        {
            "steps": len(train),
            "trajectories": trajectories,
            "routes": routes,
            "teacher_endpoint_requests": requests,
            "teacher_scored_trajectories": sum(routes.values()),
            "teacher_scoring_overrun": sum(routes.values()) - trajectories,
            "total_tokens": int(round(sum(token_counts))),
            "output_tokens": int(round(output_tokens)),
            "mean_response_tokens": output_tokens / trajectories,
            "truncation_ratio": mean([float(row["rollout/truncated_ratio"]) for row in rollout]),
            "reverse_kl_estimate": triplet([float(row["train/opd_reverse_kl"]) for row in train]),
            "native_loss": triplet([float(row["train/loss"]) for row in train]),
            "grad_norm": {
                "mean": mean([float(row["train/grad_norm"]) for row in train]),
                "max": max(float(row["train/grad_norm"]) for row in train),
            },
            "pipeline": {
                "seconds": sum(pipeline_times),
                "mean_step_seconds": mean(pipeline_times),
                "median_step_seconds": median(pipeline_times),
                "steady_median_step_seconds": median(pipeline_times[1:]),
                "trajectories_per_second": trajectories / sum(pipeline_times),
                "tokens_per_second": sum(token_counts) / sum(pipeline_times),
            },
            "components": {
                "mean_generation_seconds": mean([float(row["perf/rollout_time"]) for row in rollout]),
                "mean_train_wait_seconds": mean([float(row["perf/train_wait_time"]) for row in perf]),
                "mean_actor_train_seconds": mean([float(row["perf/actor_train_time"]) for row in perf]),
                "mean_teacher_logprob_seconds": mean(
                    [float(row["perf/ref_log_probs_time"]) for row in perf]
                ),
                "mean_student_logprob_seconds": mean([float(row["perf/log_probs_time"]) for row in perf]),
                "mean_weight_update_seconds": mean(
                    [float(row["perf/update_weights_time"]) for row in perf]
                ),
            },
            "max_off_policy_updates": 0,
        }
    )
    return result


def parse_verl_line(line: str) -> dict[str, float]:
    result = {}
    for part in line.split(" - "):
        if ":" not in part:
            continue
        key, value = part.rsplit(":", 1)
        try:
            result[key.rsplit(") ", 1)[-1]] = float(value)
        except ValueError:
            pass
    return result


def parse_verl(run_dir: Path) -> dict[str, Any]:
    result = base(run_dir, {"actor_and_rollout": [0, 1, 2, 3, 4, 5], "teachers": [6, 7]})
    text = ANSI.sub("", (run_dir / "run.log").read_text(encoding="utf-8", errors="replace"))
    rows = [
        parse_verl_line(line)
        for line in text.splitlines()
        if re.search(r"\) step:\d+ - global_seqlen/min:", line)
    ]
    check_steps("verl", rows)
    pipeline_times = [row["timing_s/step"] for row in rows]
    native_token_counts = [row["perf/total_num_tokens"] for row in rows]
    synthetic_tokens_per_step = 8 * 2
    token_counts = [value - synthetic_tokens_per_step for value in native_token_counts]
    requests = teacher_requests(run_dir, "POST /generate")
    routes = {domain: count - 1 for domain, count in requests.items()}
    trajectories = sum(routes.values())
    output_tokens = sum(row["response_length/mean"] * TRAJECTORIES_PER_STEP for row in rows)
    padding_events = re.findall(
        r"Upsampled batch from (\d+) to (\d+) with (\d+) synthetic padding samples", text
    )
    result.update(
        {
            "steps": len(rows),
            "trajectories": trajectories,
            "routes": routes,
            "teacher_endpoint_requests": requests,
            "teacher_scored_trajectories": sum(routes.values()),
            "teacher_scoring_overrun": sum(routes.values()) - trajectories,
            "total_tokens": int(sum(token_counts)),
            "output_tokens": int(round(output_tokens)),
            "mean_response_tokens": output_tokens / trajectories,
            "truncation_ratio": mean([row["response_length/clip_ratio"] for row in rows]),
            "reverse_kl_estimate": triplet([row["actor/distillation/loss"] for row in rows]),
            "native_loss": triplet([row["actor/loss"] for row in rows]),
            "grad_norm": {
                "mean": mean([row["actor/grad_norm"] for row in rows]),
                "max": max(row["actor/grad_norm"] for row in rows),
            },
            "pipeline": {
                "seconds": sum(pipeline_times),
                "mean_step_seconds": mean(pipeline_times),
                "median_step_seconds": median(pipeline_times),
                "steady_median_step_seconds": median(pipeline_times[1:]),
                "trajectories_per_second": trajectories / sum(pipeline_times),
                "tokens_per_second": sum(token_counts) / sum(pipeline_times),
            },
            "components": {
                "mean_generation_seconds": mean([row["timing_s/gen"] for row in rows]),
                "mean_old_logprob_seconds": mean([row["timing_s/old_log_prob"] for row in rows]),
                "mean_actor_update_seconds": mean([row["timing_s/update_actor"] for row in rows]),
                "mean_weight_update_seconds": mean([row["timing_s/update_weights"] for row in rows]),
                "mean_actor_mfu_pct": 100 * mean([row["perf/mfu/actor"] for row in rows]),
            },
            "padding": {
                "events": len(padding_events),
                "real_rows_per_step": 64,
                "synthetic_zero_loss_rows_per_step": 8,
                "synthetic_zero_loss_tokens_per_step": synthetic_tokens_per_step,
                "native_reported_tokens": int(sum(native_token_counts)),
            },
            "max_off_policy_updates": int(
                max(row["training/off_policy/trajectory_staleness_worst/max"] for row in rows)
            ),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        "recipe": {
            "student": "Qwen/Qwen3-8B",
            "math_teacher": "Qwen/Qwen3-32B",
            "code_teacher": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "steps": EXPECTED_STEPS,
            "prompts_per_step": 16,
            "rollouts_per_prompt": 4,
            "trajectories_per_step": TRAJECTORIES_PER_STEP,
            "total_trajectories": EXPECTED_STEPS * TRAJECTORIES_PER_STEP,
            "max_prompt_tokens": 1024,
            "max_response_tokens": 16384,
        },
        "frameworks": {
            "prime-rl": parse_prime(args.results_dir / "prime"),
            "miles": parse_miles(args.results_dir / "miles"),
            "verl": parse_verl(args.results_dir / "verl"),
        },
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
