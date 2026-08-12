#!/usr/bin/env python3
"""Build a machine-readable matched one-teacher vs two-teacher comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TEACHER_GPU = {"prime-rl": "0", "miles": "1", "verl": "1"}


def nested(row: dict[str, Any], *path: str) -> float:
    value: Any = row
    for key in path:
        value = value[key]
    return float(value)


def comparison(single: float, multi: float) -> dict[str, float]:
    return {
        "one_teacher": single,
        "two_teachers": multi,
        "absolute_delta": multi - single,
        "two_vs_one_percent": (multi / single - 1) * 100,
    }


def main() -> None:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opd", type=Path, default=root / "results-opd" / "summary.json"
    )
    parser.add_argument(
        "--mopd", type=Path, default=root / "results" / "summary.json"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "results-opd-vs-mopd.json"
    )
    args = parser.parse_args()

    opd = json.loads(args.opd.read_text(encoding="utf-8"))
    mopd = json.loads(args.mopd.read_text(encoding="utf-8"))
    if opd["recipe"] != mopd["recipe"]:
        raise ValueError("The OPD and MOPD recipe shapes do not match")

    frameworks: dict[str, Any] = {}
    for name in ("prime-rl", "miles", "verl"):
        single = opd["frameworks"][name]
        multi = mopd["frameworks"][name]
        gpu = TEACHER_GPU[name]
        frameworks[name] = {
            "active_trajectories_per_second": comparison(
                nested(single, "active", "trajectories_per_second"),
                nested(multi, "active", "trajectories_per_second"),
            ),
            "active_tokens_per_second": comparison(
                nested(single, "active", "tokens_per_second"),
                nested(multi, "active", "tokens_per_second"),
            ),
            "active_seconds": comparison(
                nested(single, "active", "seconds"),
                nested(multi, "active", "seconds"),
            ),
            "steady_median_step_seconds": comparison(
                nested(single, "active", "steady_median_step_seconds"),
                nested(multi, "active", "steady_median_step_seconds"),
            ),
            "teacher_start_seconds": comparison(
                nested(single, "wall", "teacher_start_seconds"),
                nested(multi, "wall", "teacher_start_seconds"),
            ),
            "framework_wall_seconds": comparison(
                nested(single, "wall", "framework_wall_seconds"),
                nested(multi, "wall", "framework_wall_seconds"),
            ),
            "total_wall_seconds": comparison(
                nested(single, "wall", "wall_seconds"),
                nested(multi, "wall", "wall_seconds"),
            ),
            "training_window_energy_wh": comparison(
                nested(single, "gpu", "total_energy_wh"),
                nested(multi, "gpu", "total_energy_wh"),
            ),
            "teacher_side_peak_memory_gib": {
                "gpu_index": int(gpu),
                **comparison(
                    nested(single, "gpu", "gpus", gpu, "peak_memory_mib") / 1024,
                    nested(multi, "gpu", "gpus", gpu, "peak_memory_mib") / 1024,
                ),
            },
            "mean_response_tokens": comparison(
                nested(single, "mean_response_tokens"),
                nested(multi, "mean_response_tokens"),
            ),
            "endpoint_requests": {
                "one_teacher": sum(single["teacher_endpoint_requests"].values()),
                "two_teachers": sum(multi["teacher_endpoint_requests"].values()),
            },
        }

    output = {
        "recipe": opd["recipe"],
        "comparison": "one physical math-teacher endpoint vs routed math+code endpoints",
        "controls": (
            "Same student checkpoint, prompt pool, batch shape, optimizer, loss, "
            "token limits, and nominal seed in each framework."
        ),
        "quality_equivalent": False,
        "quality_caveat": (
            "The one-teacher arm sends both logical domains to the math policy. "
            "It isolates systems footprint but is not a model-quality control."
        ),
        "repetitions_per_cell": 1,
        "frameworks": frameworks,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
