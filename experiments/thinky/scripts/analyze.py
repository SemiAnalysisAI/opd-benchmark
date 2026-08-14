#!/usr/bin/env python3
"""Reduce a Tinker OPD run and compare it with the three local frameworks."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request
from pathlib import Path
from typing import Any

PRICING_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"
CAVEATS = [
    "Tinker uses hosted LoRA training; the local Prime-RL, Miles, and verl runs use full-model updates.",
    "The original 0.6B reverse-text checkpoints are unavailable in Tinker's hosted model catalog.",
    "Hosted GPU type, utilization, energy, and queueing internals are not exposed to this client.",
    "Estimated cost is a token-price range, not an invoice; actual cache hits and billing records may differ.",
]


def numeric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty metric series")
    steady = values[1:] or values
    return {
        "initial": values[0],
        "final": values[-1],
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "steady_mean": statistics.fmean(steady),
        "tail5_mean": statistics.fmean(values[-5:]),
        "min": min(values),
        "max": max(values),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["step"]))
    return rows


def count_jsonl_records(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def read_env(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        try:
            values[key] = int(value)
        except ValueError:
            try:
                values[key] = float(value)
            except ValueError:
                values[key] = value
    return values


def load_pricing(source: str | None) -> list[dict[str, Any]] | None:
    if source is None:
        return None
    try:
        if source.startswith(("https://", "http://")):
            request = urllib.request.Request(
                source,
                headers={"User-Agent": "opd-tinker-hosted-comparison/0.1"},
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.load(response)
        return json.loads(Path(source).read_text())
    except (OSError, ValueError) as exc:  # Timing/quality remain useful offline.
        print(
            f"warning: could not load Tinker pricing from {source}: {exc}",
            file=sys.stderr,
        )
        return None


def _price(value: str) -> float:
    return float(value.removeprefix("$"))


def estimate_cost(
    student_model: str,
    teacher_model: str,
    prompt_tokens: int,
    response_tokens: int,
    pricing: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if pricing is None:
        return None
    by_id = {row["tinker_id"]: row for row in pricing}
    if student_model not in by_id or teacher_model not in by_id:
        return None
    student = by_id[student_model]
    teacher = by_id[teacher_model]
    sequence_tokens = prompt_tokens + response_tokens

    fixed = (
        response_tokens * _price(student["sample"])
        + sequence_tokens * _price(student["train"])
    ) / 1_000_000
    cached_prefill = (
        prompt_tokens * _price(student["cached_prefill"])
        + sequence_tokens * _price(teacher["cached_prefill"])
    ) / 1_000_000
    uncached_prefill = (
        prompt_tokens * _price(student["prefill"])
        + sequence_tokens * _price(teacher["prefill"])
    ) / 1_000_000
    return {
        "currency": "USD",
        "lower_bound_all_prefill_cached": fixed + cached_prefill,
        "upper_bound_no_prefill_cached": fixed + uncached_prefill,
        "pricing_source": PRICING_URL,
        "student_prices_per_million_tokens": {
            key: student[key]
            for key in ("prefill", "cached_prefill", "sample", "train")
        },
        "teacher_prices_per_million_tokens": {
            key: teacher[key] for key in ("prefill", "cached_prefill")
        },
    }


def _teacher_model(config: dict[str, Any]) -> str:
    try:
        return str(config["dataset_configs"][0]["teacher_config"]["base_model"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "config.json does not contain the Tinker teacher model"
        ) from exc


def summarize_tinker(
    result_dir: Path,
    pricing: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    metrics = read_jsonl(result_dir / "metrics.jsonl")
    config = json.loads((result_dir / "config.json").read_text())
    wall = read_env(result_dir / "wall_time.env")
    if not metrics:
        raise ValueError(f"no metrics found in {result_dir}")

    required = (
        "env/all/task_score",
        "env/all/truncated",
        "env/all/ac_tokens_per_turn",
        "env/all/total_episodes",
        "env/all/total_ob_tokens",
        "env/all/total_ac_tokens",
        "teacher_kl",
        "time/total",
    )
    for key in required:
        if any(key not in row for row in metrics):
            raise ValueError(f"Tinker metrics are missing required key {key!r}")

    total_trajectories = int(
        sum(float(row["env/all/total_episodes"]) for row in metrics)
    )
    prompt_tokens = int(sum(float(row["env/all/total_ob_tokens"]) for row in metrics))
    response_tokens = int(sum(float(row["env/all/total_ac_tokens"]) for row in metrics))
    service_step_seconds = [float(row["time/total"]) for row in metrics]
    wall_seconds = float(wall.get("wall_seconds", sum(service_step_seconds)))
    student_model = str(config["model_name"])
    teacher_model = _teacher_model(config)

    return {
        "provider": "Thinking Machines Tinker",
        "framework": "Tinker hosted OPD",
        "student_model": student_model,
        "teacher_model": teacher_model,
        "update_type": f"LoRA rank {config.get('lora_rank', 'unknown')}",
        "steps": len(metrics),
        "wall": wall,
        "wall_seconds": wall_seconds,
        "total_trajectories": total_trajectories,
        "end_to_end_trajectories_per_second": total_trajectories / wall_seconds,
        "prompt_tokens": prompt_tokens,
        "response_tokens_total": response_tokens,
        "end_to_end_response_tokens_per_second": response_tokens / wall_seconds,
        "task_score": numeric_stats(
            [float(row["env/all/task_score"]) for row in metrics]
        ),
        "truncation_rate": numeric_stats(
            [float(row["env/all/truncated"]) for row in metrics]
        ),
        "response_tokens": numeric_stats(
            [float(row["env/all/ac_tokens_per_turn"]) for row in metrics]
        ),
        "reverse_kl": numeric_stats([float(row["teacher_kl"]) for row in metrics]),
        "native_step_seconds": numeric_stats(service_step_seconds),
        "estimated_cost": estimate_cost(
            student_model,
            teacher_model,
            prompt_tokens,
            response_tokens,
            pricing,
        ),
        "checkpoint_records": count_jsonl_records(result_dir / "checkpoints.jsonl")
        if (result_dir / "checkpoints.jsonl").exists()
        else 0,
    }


def comparison_rows(
    local_summary: dict[str, Any], hosted: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for framework in local_summary["frameworks"]:
        rows.append(
            {
                "execution": "self-hosted",
                "framework": framework["framework"],
                "model": "Qwen3-0.6B reverse-text pair",
                "wall_seconds": framework["wall"]["wall_seconds"],
                "trajectories_per_second": framework[
                    "end_to_end_trajectories_per_second"
                ],
                "task_score_final": framework["task_score"]["final"],
                "task_score_tail5": framework["task_score"]["tail5_mean"],
                "reverse_kl_final": framework["reverse_kl"]["final"],
                "energy_wh": framework["gpu"]["combined_energy_wh"],
                "estimated_cost_usd": None,
            }
        )
    cost = hosted["estimated_cost"]
    rows.append(
        {
            "execution": "hosted",
            "framework": hosted["framework"],
            "model": f"{hosted['student_model']} <- {hosted['teacher_model']}",
            "wall_seconds": hosted["wall_seconds"],
            "trajectories_per_second": hosted["end_to_end_trajectories_per_second"],
            "task_score_final": hosted["task_score"]["final"],
            "task_score_tail5": hosted["task_score"]["tail5_mean"],
            "reverse_kl_final": hosted["reverse_kl"]["final"],
            "energy_wh": None,
            "estimated_cost_usd": (
                [
                    cost["lower_bound_all_prefill_cached"],
                    cost["upper_bound_no_prefill_cached"],
                ]
                if cost
                else None
            ),
        }
    )
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# OPD execution comparison",
        "",
        "| Execution | Framework | Model pair | Wall (s) | Traj/s | Final task | Tail-5 task | Final reverse KL | Energy (Wh) | Estimated cost (USD) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cost = row["estimated_cost_usd"]
        cost_text = f"{cost[0]:.4f}–{cost[1]:.4f}" if cost else "—"
        lines.append(
            "| {execution} | {framework} | {model} | {wall} | {throughput} | {final_task} | "
            "{tail_task} | {final_kl} | {energy} | {cost} |".format(
                execution=row["execution"],
                framework=row["framework"],
                model=row["model"],
                wall=_fmt(row["wall_seconds"], 1),
                throughput=_fmt(row["trajectories_per_second"]),
                final_task=_fmt(row["task_score_final"]),
                tail_task=_fmt(row["task_score_tail5"]),
                final_kl=_fmt(row["reverse_kl_final"]),
                energy=_fmt(row["energy_wh"], 2),
                cost=cost_text,
            )
        )
    lines.extend(["", "## Comparability limits", ""])
    lines.extend(f"- {caveat}" for caveat in CAVEATS)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tinker-results", type=Path, required=True)
    parser.add_argument(
        "--local-summary",
        type=Path,
        default=repo_root
        / "benchmarks"
        / "opd-frameworks"
        / "results"
        / "summary.json",
    )
    parser.add_argument("--pricing-json", default=PRICING_URL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.tinker_results.resolve()
    pricing = load_pricing(args.pricing_json)
    hosted = summarize_tinker(result_dir, pricing)
    local_summary = json.loads(args.local_summary.read_text())
    rows = comparison_rows(local_summary, hosted)
    output = args.output or result_dir / "hosted_comparison.json"
    markdown_output = args.markdown_output or result_dir / "hosted_comparison.md"
    payload = {
        "hosted": hosted,
        "comparison_rows": rows,
        "comparability_limits": CAVEATS,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    markdown_output.write_text(render_markdown(rows))
    print(output)
    print(markdown_output)


if __name__ == "__main__":
    main()
