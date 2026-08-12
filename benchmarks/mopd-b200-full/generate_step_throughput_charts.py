#!/usr/bin/env python3
"""Generate per-step B200 throughput charts from the native framework logs."""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import re
import shutil
import subprocess
from pathlib import Path


WIDTH = 1400
HEIGHT = 720
FONT = "Inter, Arial, Helvetica, sans-serif"
TRAJECTORIES_PER_STEP = 64
ANSI = re.compile(r"\x1b\[[0-9;]*m")
THROUGHPUT_COLOR = "#245A8D"
SIGNAL_COLOR = "#9A5B43"
REFERENCE_COLOR = "#526176"
LABELS = {"prime-rl": "Prime-RL", "miles": "Miles", "verl": "verl"}


def text(
    x: float,
    y: float,
    value: object,
    size: int,
    *,
    fill: str = "#15263F",
    weight: int = 400,
    anchor: str = "start",
    rotate: int | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{transform}>'
        f"{html.escape(str(value))}</text>"
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = "#D9E0E9",
    width: float = 1,
    dash: str | None = None,
) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dashed}/>'
    )


def parse_prime(results: Path) -> list[tuple[int, float, int, float]]:
    trainer: dict[int, dict[str, object]] = {}
    for raw in (results / "prime" / "trainer-metrics.jsonl").read_text().splitlines():
        row = json.loads(raw)
        trainer.setdefault(int(row["step"]), {}).update(row)
    rows = []
    for raw in (results / "prime" / "orchestrator-metrics.jsonl").read_text().splitlines():
        row = json.loads(raw)
        if "progress/tokens" in row:
            step = int(row["step"])
            rows.append(
                (
                    step,
                    float(row["time/step"]),
                    int(row["progress/rollouts"]),
                    -float(trainer[step]["ref_kl/mean"]),
                )
            )
    return sorted(rows)


def parse_miles(results: Path) -> list[tuple[int, float, int, float]]:
    log = ANSI.sub("", (results / "miles" / "run.log").read_text(errors="replace"))
    perf = {
        int(match.group(1)): ast.literal_eval(match.group(2))
        for match in re.finditer(r"train_metric_utils\.py:50 - perf (\d+): (\{[^\n]+\})", log)
    }
    train = {
        int(match.group(1)): ast.literal_eval(match.group(2))
        for match in re.finditer(r"log_utils\.py:544 - step (\d+): (\{[^\n]+\})", log)
    }
    # Miles numbers native steps from 0; normalize the published x-axis to 1–15.
    return [
        (
            published_step,
            float(row["perf/step_time"]),
            TRAJECTORIES_PER_STEP,
            float(train[native_step]["train/opd_reverse_kl"]),
        )
        for published_step, (native_step, row) in enumerate(sorted(perf.items()), start=1)
    ]


def parse_verl(results: Path) -> list[tuple[int, float, int, float]]:
    log = ANSI.sub("", (results / "verl" / "run.log").read_text(errors="replace"))
    rows = []
    for raw in log.splitlines():
        if not re.search(r"\) step:\d+ - global_seqlen/min:", raw):
            continue
        values: dict[str, float] = {}
        for part in raw.split(" - "):
            if ":" not in part:
                continue
            key, value = part.rsplit(":", 1)
            try:
                values[key.rsplit(") ", 1)[-1]] = float(value)
            except ValueError:
                pass
        rows.append(
            (
                int(values["step"]),
                values["timing_s/step"],
                TRAJECTORIES_PER_STEP,
                values["actor/distillation/loss"],
            )
        )
    return sorted(rows)


def build_svg(name: str, rows: list[tuple[int, float, int, float]]) -> str:
    if len(rows) != 15:
        raise ValueError(f"{name}: expected 15 steps, found {len(rows)}")

    label = LABELS[name]
    points = [(step, trajectories / seconds, signal) for step, seconds, trajectories, signal in rows]
    aggregate_rate = sum(trajectories for _, _, trajectories, _ in rows) / sum(
        seconds for _, seconds, _, _ in rows
    )
    y_max = max(3, math.ceil(max(value for _, value, _ in points) * 1.12))
    signal_max = 0.8

    left, right = 122, WIDTH - 54
    top, bottom = 156, 600
    plot_width = right - left
    plot_height = bottom - top

    def x(step: int) -> float:
        return left + (step - 1) / 14 * plot_width

    def y(value: float) -> float:
        return bottom - value / y_max * plot_height

    def signal_y(value: float) -> float:
        return bottom - value / signal_max * plot_height

    title = f"{label} B200 throughput and OPD signal by step"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        '<desc id="desc">Real trajectories per second and mean sampled reverse KL for each '
        "of fifteen optimizer steps. Task reward is disabled and remains zero.</desc>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#F8FAFC"/>',
        text(54, 58, title, 36, weight=700),
        text(
            54,
            94,
            "Throughput on left axis · mean sampled reverse KL on right axis · task reward = 0",
            18,
            fill="#607086",
            weight=500,
        ),
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        'fill="#FFFFFF" stroke="#C9D2DF" stroke-width="1.2"/>',
    ]

    for tick in range(y_max + 1):
        yy = y(tick)
        parts.extend(
            [
                line(left, yy, right, yy, stroke="#DEE4EC", dash="4 7" if tick else None),
                text(left - 18, yy + 5, tick, 14, fill="#69798E", anchor="end"),
            ]
        )

    for tick in range(5):
        value = tick * 0.2
        yy = signal_y(value)
        parts.extend(
            [
                line(right, yy, right + 7, yy, stroke="#8795A8"),
                text(right + 17, yy + 5, f"{value:.1f}", 14, fill="#69798E"),
            ]
        )

    for step in range(1, 16):
        xx = x(step)
        parts.extend(
            [
                line(xx, bottom, xx, bottom + 7, stroke="#8795A8"),
                text(xx, bottom + 30, step, 14, fill="#607086", anchor="middle"),
            ]
        )

    average_y = y(aggregate_rate)
    parts.extend(
        [
            line(left, average_y, right, average_y, stroke=REFERENCE_COLOR, width=2, dash="9 7"),
            line(left, 124, left + 52, 124, stroke=THROUGHPUT_COLOR, width=4),
            text(left + 64, 130, "throughput", 15, fill="#45566D", weight=600),
            line(left + 196, 124, left + 248, 124, stroke=SIGNAL_COLOR, width=3, dash="7 5"),
            text(left + 260, 130, "mean reverse KL", 15, fill="#45566D", weight=600),
            line(right - 282, 124, right - 224, 124, stroke=REFERENCE_COLOR, width=2, dash="9 7"),
            text(right - 210, 130, f"aggregate {aggregate_rate:.2f} traj/s", 15, fill="#45566D", weight=600),
        ]
    )

    path = " ".join(
        ("M" if index == 0 else "L") + f" {x(step):.1f} {y(value):.1f}"
        for index, (step, value, _) in enumerate(points)
    )
    signal_path = " ".join(
        ("M" if index == 0 else "L") + f" {x(step):.1f} {signal_y(signal):.1f}"
        for index, (step, _, signal) in enumerate(points)
    )
    area = path + f" L {x(points[-1][0]):.1f} {bottom:.1f} L {x(points[0][0]):.1f} {bottom:.1f} Z"
    parts.extend(
        [
            f'<path d="{area}" fill="{THROUGHPUT_COLOR}" opacity="0.08"/>',
            f'<path d="{path}" fill="none" stroke="{THROUGHPUT_COLOR}" stroke-width="4" '
            'stroke-linejoin="round" stroke-linecap="round"/>',
            f'<path d="{signal_path}" fill="none" stroke="{SIGNAL_COLOR}" stroke-width="3" '
            'stroke-dasharray="7 5" stroke-linejoin="round" stroke-linecap="round"/>',
        ]
    )

    for step, value, signal in points:
        xx, yy = x(step), y(value)
        label_y = yy - 13 if yy > top + 34 else yy + 28
        parts.extend(
            [
                f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="6" fill="#FFFFFF" '
                f'stroke="{THROUGHPUT_COLOR}" stroke-width="4"/>',
                text(xx, label_y, f"{value:.2f}", 13, fill="#263A55", weight=600, anchor="middle"),
                f'<rect x="{xx - 4.5:.1f}" y="{signal_y(signal) - 4.5:.1f}" width="9" height="9" '
                f'fill="#FFFFFF" stroke="{SIGNAL_COLOR}" stroke-width="3"/>',
            ]
        )

    parts.extend(
        [
            text((left + right) / 2, 674, "Optimizer step", 16, fill="#4F6077", weight=600, anchor="middle"),
            text(38, (top + bottom) / 2, "Real trajectories / second", 16, fill="#4F6077", weight=600, anchor="middle", rotate=-90),
            text(WIDTH - 18, (top + bottom) / 2, "Mean sampled reverse KL", 16, fill="#4F6077", weight=600, anchor="middle", rotate=90),
            text(
                right,
                704,
                "Source: native framework logs · task reward disabled · one run",
                13,
                fill="#798698",
                anchor="end",
            ),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def write_chart(output_dir: Path, name: str, rows: list[tuple[int, float, int, float]]) -> None:
    svg = output_dir / f"{name}-b200-throughput-by-step.svg"
    png = output_dir / f"{name}-b200-throughput-by-step.png"
    svg.write_text(build_svg(name, rows), encoding="utf-8")
    renderer = shutil.which("sips")
    if renderer:
        subprocess.run(
            [renderer, "-s", "format", "png", str(svg), "--out", str(png)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    print(svg)
    if png.exists():
        print(png)


def main() -> None:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--output-dir", type=Path, default=root / "charts")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series = {
        "prime-rl": parse_prime(args.results),
        "miles": parse_miles(args.results),
        "verl": parse_verl(args.results),
    }
    for name, rows in series.items():
        write_chart(args.output_dir, name, rows)


if __name__ == "__main__":
    main()
