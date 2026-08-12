#!/usr/bin/env python3
"""Parse the three framework-native logs into comparable benchmark artifacts."""

from __future__ import annotations

import ast
import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STEPS = 15
TRAJECTORIES_PER_STEP = 128
TOTAL_TRAJECTORIES = STEPS * TRAJECTORIES_PER_STEP
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def env_file(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        values[key] = int(value) if value.isdigit() else value
    return values


def merged_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        item = json.loads(line)
        rows[int(item["step"])].update(item)
    return dict(rows)


def stats(values: list[float], *, steady_from: int = 1) -> dict[str, float]:
    steady = values[steady_from:]
    return {
        "initial": values[0],
        "final": values[-1],
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "steady_mean": statistics.fmean(steady),
        "steady_median": statistics.median(steady),
        "tail5_mean": statistics.fmean(values[-5:]),
        "min": min(values),
        "max": max(values),
    }


def gpu_stats(path: Path) -> dict[str, Any]:
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            index = int(raw["index"].strip())
            rows[index].append(
                {
                    "timestamp": datetime.strptime(raw["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f"),
                    "utilization_pct": float(raw["utilization_gpu_pct"].strip()),
                    "memory_mib": float(raw["memory_used_mib"].strip()),
                    "power_w": float(raw["power_w"].strip()),
                }
            )

    result: dict[str, Any] = {}
    total_energy = 0.0
    for index, samples in sorted(rows.items()):
        energy_wh = 0.0
        for left, right in zip(samples, samples[1:]):
            seconds = (right["timestamp"] - left["timestamp"]).total_seconds()
            energy_wh += seconds * (left["power_w"] + right["power_w"]) / 2 / 3600
        total_energy += energy_wh
        result[str(index)] = {
            "samples": len(samples),
            "mean_utilization_pct": statistics.fmean(x["utilization_pct"] for x in samples),
            "p95_utilization_pct": sorted(x["utilization_pct"] for x in samples)[int(0.95 * (len(samples) - 1))],
            "peak_memory_gib": max(x["memory_mib"] for x in samples) / 1024,
            "mean_power_w": statistics.fmean(x["power_w"] for x in samples),
            "energy_wh": energy_wh,
        }
    result["combined_energy_wh"] = total_energy
    return result


def parse_prime() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folder = RESULTS / "prime"
    wall = env_file(folder / "wall_time.env")
    trainer = merged_jsonl(folder / "trainer_metrics.jsonl")
    rollout = merged_jsonl(folder / "metrics.jsonl")
    ordered = range(1, STEPS + 1)

    rows = []
    for step in ordered:
        train = trainer[step]
        roll = rollout[step]
        rows.append(
            {
                "framework": "Prime-RL",
                "step": step,
                "task_score": roll["train/agg/effective/agent/reward/mean"],
                "truncation_rate": roll["train/agg/effective/agent/is_truncated/mean"],
                "response_tokens": roll["train/agg/effective/agent/num_output_tokens/mean"],
                "reverse_kl": -train["ref_kl/mean"],
                "native_step_seconds": train["time/step"],
                "native_tokens_per_second": train["perf/throughput"],
                "actor_mfu_pct": train["perf/mfu"],
                "grad_norm": train["optim/grad_norm"],
            }
        )

    first_update_epoch = trainer[1]["time"]
    summary = {
        "framework": "Prime-RL",
        "wall": wall,
        "time_to_first_update_seconds": first_update_epoch - wall["start_epoch"],
        "task_score": stats([x["task_score"] for x in rows]),
        "truncation_rate": stats([x["truncation_rate"] for x in rows]),
        "response_tokens": stats([x["response_tokens"] for x in rows]),
        "reverse_kl": stats([x["reverse_kl"] for x in rows]),
        "native_step_seconds": stats([x["native_step_seconds"] for x in rows]),
        "native_tokens_per_second": stats([x["native_tokens_per_second"] for x in rows]),
        "actor_mfu_pct": stats([x["actor_mfu_pct"] for x in rows]),
        "peak_actor_memory_gib_native": max(trainer[s]["perf/peak_memory"] for s in ordered),
        "gpu": gpu_stats(folder / "gpu.csv"),
    }
    return summary, rows


def parse_verl() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folder = RESULTS / "verl"
    wall = env_file(folder / "wall_time.env")
    rows = []
    for raw in (folder / "run.log").read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw)
        if "TaskRunnerV1" not in line or " step:" not in line or "training/global_step:" not in line:
            continue
        fields: dict[str, float] = {}
        payload = line[line.index("step:") :]
        for part in payload.split(" - "):
            key, value = part.split(":", 1)
            try:
                fields[key] = float(value)
            except ValueError:
                pass
        rows.append(
            {
                "framework": "verl",
                "step": int(fields["step"]),
                "task_score": fields["critic/score/mean"],
                "truncation_rate": fields["response_length/clip_ratio"],
                "response_tokens": fields["response_length/mean"],
                "reverse_kl": fields["actor/distillation/loss"],
                "native_step_seconds": fields["timing_s/step"],
                "native_tokens_per_second": fields["perf/throughput"],
                "actor_mfu_pct": fields["perf/mfu/actor"] * 100,
                "grad_norm": fields["actor/grad_norm"],
                "peak_actor_memory_gib": fields["actor/perf/max_memory_allocated_gb"],
            }
        )
    rows.sort(key=lambda x: x["step"])
    if len(rows) != STEPS:
        raise ValueError(f"expected {STEPS} verl steps, found {len(rows)}")

    native_step_sum = sum(x["native_step_seconds"] for x in rows)
    initialization_approx = wall["wall_seconds"] - native_step_sum
    summary = {
        "framework": "verl",
        "wall": wall,
        "time_to_first_update_seconds_approx": initialization_approx + rows[0]["native_step_seconds"],
        "initialization_seconds_approx": initialization_approx,
        "task_score": stats([x["task_score"] for x in rows]),
        "truncation_rate": stats([x["truncation_rate"] for x in rows]),
        "response_tokens": stats([x["response_tokens"] for x in rows]),
        "reverse_kl": stats([x["reverse_kl"] for x in rows]),
        "native_step_seconds": stats([x["native_step_seconds"] for x in rows]),
        "native_tokens_per_second": stats([x["native_tokens_per_second"] for x in rows]),
        "actor_mfu_pct": stats([x["actor_mfu_pct"] for x in rows]),
        "peak_actor_memory_gib_native": max(x["peak_actor_memory_gib"] for x in rows),
        "gpu": gpu_stats(folder / "gpu.csv"),
    }
    return summary, rows


def dict_records(path: Path, marker: str) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw)
        if marker not in line:
            continue
        match = re.search(r"(?:step|perf|rollout) (\d+): (\{.*\})$", line)
        if match:
            records.append((int(match.group(1)), ast.literal_eval(match.group(2))))
    return records


def parse_miles() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folder = RESULTS / "miles"
    wall = env_file(folder / "wall_time.env")
    path = folder / "run.log"
    train = dict(dict_records(path, "log_utils.py:544 - step"))
    perf = dict(dict_records(path, "train_metric_utils.py:50 - perf"))
    rollout = dict(dict_records(path, "log_utils.py:123 - rollout"))

    scores: list[tuple[int, float, float, int]] = []
    for line in path.read_text(errors="replace").splitlines():
        match = re.search(
            r"OPD_BENCH_TASK_SCORE count=(\d+) mean=([0-9.]+) truncation_rate=([0-9.]+)",
            ANSI.sub("", line),
        )
        if match:
            scores.append((len(scores), float(match.group(2)), float(match.group(3)), int(match.group(1))))

    rows = []
    for step in range(STEPS):
        task_score = scores[step][1] if len(scores) == STEPS else None
        score_truncation = scores[step][2] if len(scores) == STEPS else None
        rows.append(
            {
                "framework": "Miles",
                "step": step + 1,
                "task_score": task_score,
                "truncation_rate": score_truncation if score_truncation is not None else rollout[step]["rollout/truncated"],
                "response_tokens": rollout[step]["rollout/response_lengths"],
                "reverse_kl": train[step]["train/opd_reverse_kl"],
                "native_step_seconds": perf[step]["perf/step_time"],
                "native_tokens_per_second": perf[step]["perf/actor_train_tok_per_s"],
                "actor_mfu_pct": None,
                "grad_norm": train[step]["train/grad_norm"],
                "wait_ratio": perf[step]["perf/wait_time_ratio"],
            }
        )

    first_line = next(
        line for line in path.read_text(errors="replace").splitlines() if "log_utils.py:544 - step 0:" in line
    )
    first_timestamp = datetime.strptime(
        re.search(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) ", first_line).group(1),
        "%Y-%m-%d %H:%M:%S.%f",
    ).replace(tzinfo=timezone.utc)
    start_timestamp = datetime.fromtimestamp(wall["start_epoch"], tz=timezone.utc)
    summary = {
        "framework": "Miles",
        "wall": wall,
        "time_to_first_update_seconds": (first_timestamp - start_timestamp).total_seconds(),
        "task_score": stats([x["task_score"] for x in rows]) if len(scores) == STEPS else None,
        "task_score_sample_count_per_step": scores[0][3] if len(scores) == STEPS else 0,
        "truncation_rate": stats([x["truncation_rate"] for x in rows]),
        "response_tokens": stats([x["response_tokens"] for x in rows]),
        "reverse_kl": stats([x["reverse_kl"] for x in rows]),
        "native_step_seconds": stats([x["native_step_seconds"] for x in rows]),
        "native_tokens_per_second": stats([x["native_tokens_per_second"] for x in rows]),
        "wait_ratio": stats([x["wait_ratio"] for x in rows]),
        "gpu": gpu_stats(folder / "gpu.csv"),
    }
    return summary, rows


def main() -> None:
    summaries = []
    step_rows = []
    for parser in (parse_prime, parse_verl, parse_miles):
        summary, rows = parser()
        wall = summary["wall"]
        summary["total_trajectories"] = TOTAL_TRAJECTORIES
        summary["end_to_end_trajectories_per_second"] = TOTAL_TRAJECTORIES / wall["wall_seconds"]
        summary["framework_window_trajectories_per_second"] = TOTAL_TRAJECTORIES / wall["framework_wall_seconds"]
        summary["total_generated_response_tokens"] = summary["response_tokens"]["mean"] * TOTAL_TRAJECTORIES
        summary["end_to_end_generated_response_tokens_per_second"] = (
            summary["total_generated_response_tokens"] / wall["wall_seconds"]
        )
        summaries.append(summary)
        step_rows.extend(rows)

    output = {
        "recipe": {
            "steps": STEPS,
            "prompts_per_step": 8,
            "rollouts_per_prompt": 16,
            "trajectories_per_step": TRAJECTORIES_PER_STEP,
            "total_trajectories": TOTAL_TRAJECTORIES,
        },
        "frameworks": summaries,
    }
    (RESULTS / "summary.json").write_text(json.dumps(output, indent=2) + "\n")

    fieldnames = list(step_rows[0])
    for row in step_rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (RESULTS / "step_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(step_rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
