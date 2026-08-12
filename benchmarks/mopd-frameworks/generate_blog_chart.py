#!/usr/bin/env python3
"""Generate the publication chart for the MOPD framework benchmark."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


WIDTH = 1600
HEIGHT = 900
FONT = "Inter, Arial, Helvetica, sans-serif"
COLORS = {
    "verl": "#246BFD",
    "prime-rl": "#DF7726",
    "miles": "#7A5AF8",
}
LABELS = {"verl": "verl", "prime-rl": "Prime-RL", "miles": "Miles"}
STACKS = {
    "verl": "vLLM + FSDP",
    "prime-rl": "vLLM + native trainer",
    "miles": "SGLang + Megatron",
}


def text(
    x: float,
    y: float,
    value: str,
    size: int,
    *,
    fill: str = "#132238",
    weight: int = 400,
    anchor: str = "start",
    spacing: float = 0,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}" letter-spacing="{spacing}">'
        f"{html.escape(value)}</text>"
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    *,
    radius: float = 0,
    stroke: str | None = None,
    stroke_width: float = 1,
) -> str:
    border = (
        f' stroke="{stroke}" stroke-width="{stroke_width}"' if stroke else ""
    )
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
        f'height="{height:.1f}" rx="{radius:.1f}" fill="{fill}"{border}/>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = "#DDE3EA",
    width: float = 1,
    dash: str | None = None,
) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dashed}/>'
    )


def build_chart(summary: dict, output_scale: float = 1) -> str:
    frameworks = summary["frameworks"]
    order = ["verl", "prime-rl", "miles"]
    throughput = {
        name: frameworks[name]["active"]["trajectories_per_second"] for name in order
    }
    framework_wall = {
        name: frameworks[name]["wall"]["framework_wall_seconds"] for name in order
    }
    total_wall = {name: frameworks[name]["wall"]["wall_seconds"] for name in order}
    setup_wall = {name: total_wall[name] - framework_wall[name] for name in order}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH * output_scale:g}" '
        f'height="{HEIGHT * output_scale:g}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        'aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">Same MOPD recipe, different systems throughput</title>',
        '<desc id="chart-desc">Two-panel comparison of active training throughput and '
        'wall time for verl, Prime-RL, and Miles on two H100 GPUs.</desc>',
        rect(0, 0, WIDTH, HEIGHT, "#F5F7FA"),
        rect(58, 30, 165, 32, "#E5EDFF", radius=16),
        text(140, 52, "MOPD BENCHMARK", 13, fill="#2457B8", weight=700, anchor="middle", spacing=1.4),
        text(60, 116, "Same recipe, different throughput", 47, weight=750),
        text(60, 158, "Prime-RL vs Miles vs verl on 2× H100 80 GB", 22, fill="#58677C", weight=450),
        text(1540, 52, "AUGUST 2026", 13, fill="#6C7889", weight=650, anchor="end", spacing=1.2),
        line(60, 185, 1540, 185, stroke="#CDD5DF", width=1.3),
        # Cards.
        rect(60, 215, 720, 575, "#FFFFFF", radius=20, stroke="#E1E6ED"),
        rect(820, 215, 720, 575, "#FFFFFF", radius=20, stroke="#E1E6ED"),
        text(92, 264, "ACTIVE TRAINING THROUGHPUT", 14, fill="#66758A", weight=700, spacing=1.1),
        text(92, 302, "Trained trajectories per second", 25, weight=700),
        text(92, 331, "Higher is better · native active-step timers", 16, fill="#718096"),
        text(852, 264, "END-TO-END WALL TIME", 14, fill="#66758A", weight=700, spacing=1.1),
        text(852, 302, "Cold setup + framework window", 25, weight=700),
        text(852, 331, "Lower is better · seconds", 16, fill="#718096"),
    ]

    # Left panel: active throughput.
    left_plot_x = 250
    left_plot_w = 450
    left_scale = left_plot_w / 12
    for tick in (0, 4, 8, 12):
        x = left_plot_x + tick * left_scale
        parts.extend(
            [
                line(x, 370, x, 660, stroke="#E6EAF0", dash="4 7"),
                text(x, 691, str(tick), 14, fill="#7A8798", anchor="middle"),
            ]
        )

    row_y = {"verl": 395, "prime-rl": 490, "miles": 585}
    for name in order:
        y = row_y[name]
        value = throughput[name]
        bar_w = value * left_scale
        parts.extend(
            [
                text(92, y + 20, LABELS[name], 19, weight=700),
                text(92, y + 43, STACKS[name], 12, fill="#7B8797"),
                rect(left_plot_x, y, left_plot_w, 48, "#EDF1F5", radius=7),
                rect(left_plot_x, y, bar_w, 48, COLORS[name], radius=7),
                text(left_plot_x + bar_w + 13, y + 32, f"{value:.2f}", 19, weight=750),
            ]
        )

    prime_gain = throughput["verl"] / throughput["prime-rl"] - 1
    miles_gain = throughput["verl"] / throughput["miles"] - 1
    parts.extend(
        [
            rect(92, 716, 656, 48, "#EEF4FF", radius=10),
            text(112, 747, "verl", 17, fill=COLORS["verl"], weight=750),
            text(
                150,
                747,
                f"is {prime_gain:.0%} faster than Prime-RL and {miles_gain:.0%} faster than Miles",
                16,
                fill="#29415F",
                weight=550,
            ),
        ]
    )

    # Right panel: setup and framework wall time.
    right_plot_x = 1010
    right_plot_w = 455
    right_scale = right_plot_w / 350
    for tick in (0, 100, 200, 300):
        x = right_plot_x + tick * right_scale
        parts.extend(
            [
                line(x, 402, x, 660, stroke="#E6EAF0", dash="4 7"),
                text(x, 691, str(tick), 14, fill="#7A8798", anchor="middle"),
            ]
        )

    parts.extend(
        [
            rect(1217, 350, 16, 16, "#D7DEE8", radius=3),
            text(1242, 363, "teacher/setup", 13, fill="#68778A"),
            rect(1360, 350, 16, 16, COLORS["verl"], radius=3),
            text(1385, 363, "framework", 13, fill="#68778A"),
        ]
    )

    for name in order:
        y = row_y[name]
        setup_w = setup_wall[name] * right_scale
        framework_w = framework_wall[name] * right_scale
        parts.extend(
            [
                text(852, y + 29, LABELS[name], 18, weight=700),
                rect(right_plot_x, y, setup_w, 48, "#D7DEE8", radius=7),
                rect(right_plot_x + setup_w, y, framework_w, 48, COLORS[name], radius=7),
                text(
                    right_plot_x + setup_w + framework_w + 11,
                    y + 32,
                    f"{total_wall[name]:.0f}s",
                    18,
                    weight=750,
                ),
                text(
                    right_plot_x + setup_w / 2,
                    y + 31,
                    f"{setup_wall[name]:.0f}",
                    13,
                    fill="#526175",
                    weight=650,
                    anchor="middle",
                ),
            ]
        )
        if framework_w > 80:
            parts.append(
                text(
                    right_plot_x + setup_w + framework_w / 2,
                    y + 31,
                    f"{framework_wall[name]:.0f}",
                    13,
                    fill="#FFFFFF",
                    weight=700,
                    anchor="middle",
                )
            )

    vs_prime = 1 - total_wall["verl"] / total_wall["prime-rl"]
    vs_miles = 1 - total_wall["verl"] / total_wall["miles"]
    parts.extend(
        [
            rect(852, 716, 656, 48, "#EEF4FF", radius=10),
            text(872, 747, "verl total wall", 16, fill=COLORS["verl"], weight=750),
            text(
                983,
                747,
                f"is {vs_prime:.0%} below Prime-RL and {vs_miles:.0%} below Miles",
                16,
                fill="#29415F",
                weight=550,
            ),
        ]
    )

    # Footer.
    parts.extend(
        [
            line(60, 822, 1540, 822, stroke="#CDD5DF", width=1.2),
            text(
                60,
                857,
                "Qwen2.5 0.5B student · two 1.5B specialist teachers · 15 steps · 960 trained trajectories",
                15,
                fill="#58677C",
                weight=550,
            ),
            text(
                1540,
                857,
                "Single run · sampled-token reverse KL (top_k=0)",
                14,
                fill="#728094",
                anchor="end",
            ),
            text(1540, 882, "Source: MOPD framework benchmark, 2026-08-07", 12, fill="#8A95A5", anchor="end"),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).parent / "results" / "summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "charts" / "mopd-framework-benchmark.svg",
    )
    parser.add_argument("--scale", type=float, default=1)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_chart(summary, args.scale), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
