#!/usr/bin/env python3
"""Generate blog charts for the OPD/MOPD ablation and routing probe."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


FONT = "Inter, Arial, Helvetica, sans-serif"
INK = "#132238"
MUTED = "#66758A"
GRID = "#E5EAF0"
BLUE = "#246BFD"
GRAY = "#AEB9C7"


def text(
    x: float,
    y: float,
    value: str,
    size: int,
    *,
    fill: str = INK,
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
) -> str:
    border = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
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
    stroke: str = GRID,
    width: float = 1,
    dash: str | None = None,
) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dashed}/>'
    )


def svg_open(width: int, height: int, scale: float, title_value: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width * scale:g}" '
        f'height="{height * scale:g}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title_value)}</title>",
        rect(0, 0, width, height, "#F5F7FA"),
    ]


def ablation_chart(data: dict, scale: float) -> str:
    width, height = 1600, 900
    order = ["prime-rl", "miles", "verl"]
    labels = {"prime-rl": "Prime-RL", "miles": "Miles", "verl": "verl"}
    panels = [
        (
            60,
            "ACTIVE THROUGHPUT",
            "Trained trajectories / second",
            "active_trajectories_per_second",
            12.0,
            lambda value: f"{value:.2f}",
            "Higher is better",
        ),
        (
            560,
            "END-TO-END WALL TIME",
            "Setup + framework window",
            "total_wall_seconds",
            350.0,
            lambda value: f"{value:.0f}s",
            "Lower is better",
        ),
        (
            1060,
            "TEACHER-SIDE PEAK HBM",
            "Peak on the teacher-host GPU",
            "teacher_side_peak_memory_gib",
            80.0,
            lambda value: f"{value:.1f}",
            "GiB · lower is better",
        ),
    ]
    parts = svg_open(width, height, scale, "Matched OPD versus MOPD systems ablation")
    parts.extend(
        [
            rect(58, 30, 184, 32, "#E5EDFF", radius=16),
            text(150, 52, "POLICY ABLATION", 13, fill="#2457B8", weight=700, anchor="middle", spacing=1.4),
            text(60, 116, "What does the second teacher cost?", 47, weight=750),
            text(60, 158, "Matched one-policy vs two-policy OPD · same 15 × 64 training shape", 22, fill="#58677C", weight=450),
            text(1540, 52, "2× H100 80 GB", 13, fill="#6C7889", weight=650, anchor="end", spacing=1.2),
            line(60, 185, 1540, 185, stroke="#CDD5DF", width=1.3),
        ]
    )

    row_y = [414, 526, 638]
    for panel_x, kicker, title_value, metric, maximum, formatter, direction in panels:
        parts.extend(
            [
                rect(panel_x, 215, 480, 575, "#FFFFFF", radius=20, stroke="#E1E6ED"),
                text(panel_x + 30, 260, kicker, 13, fill=MUTED, weight=700, spacing=1.0),
                text(panel_x + 30, 298, title_value, 22, weight=700),
                text(panel_x + 30, 326, direction, 14, fill="#718096"),
                rect(panel_x + 30, 350, 16, 16, GRAY, radius=3),
                text(panel_x + 54, 363, "one teacher", 13, fill=MUTED),
                rect(panel_x + 158, 350, 16, 16, BLUE, radius=3),
                text(panel_x + 182, 363, "two teachers", 13, fill=MUTED),
            ]
        )
        plot_x = panel_x + 148
        plot_width = 250
        for fraction in (0, 0.5, 1):
            x = plot_x + plot_width * fraction
            parts.extend(
                [
                    line(x, 393, x, 704, dash="4 7"),
                    text(x, 726, f"{maximum * fraction:g}", 12, fill="#7A8798", anchor="middle"),
                ]
            )
        for y, name in zip(row_y, order):
            values = data["frameworks"][name][metric]
            one = float(values["one_teacher"])
            two = float(values["two_teachers"])
            one_width = one / maximum * plot_width
            two_width = two / maximum * plot_width
            parts.extend(
                [
                    text(panel_x + 30, y + 31, labels[name], 17, weight=700),
                    rect(plot_x, y, plot_width, 18, "#EEF1F5", radius=5),
                    rect(plot_x, y, one_width, 18, GRAY, radius=5),
                    rect(plot_x, y + 27, plot_width, 18, "#EEF1F5", radius=5),
                    rect(plot_x, y + 27, two_width, 18, BLUE, radius=5),
                    text(panel_x + 460, y + 14, formatter(one), 13, fill="#536277", weight=650, anchor="end"),
                    text(panel_x + 460, y + 42, formatter(two), 13, fill=BLUE, weight=750, anchor="end"),
                ]
            )

    prime = data["frameworks"]["prime-rl"]
    miles = data["frameworks"]["miles"]
    verl = data["frameworks"]["verl"]
    throughput_note = (
        f"MOPD delta: Prime {prime['active_trajectories_per_second']['two_vs_one_percent']:+.0f}% · "
        f"Miles {miles['active_trajectories_per_second']['two_vs_one_percent']:+.0f}% · "
        f"verl {verl['active_trajectories_per_second']['two_vs_one_percent']:+.0f}%"
    )
    wall_note = (
        f"Second teacher: Prime {prime['total_wall_seconds']['absolute_delta']:+.0f}s · "
        f"Miles {miles['total_wall_seconds']['absolute_delta']:+.0f}s · "
        f"verl {verl['total_wall_seconds']['absolute_delta']:+.0f}s"
    )
    memory_deltas = [
        data["frameworks"][name]["teacher_side_peak_memory_gib"]["absolute_delta"]
        for name in order
    ]
    memory_note = f"Adds {min(memory_deltas):.1f}–{max(memory_deltas):.1f} GiB peak HBM"
    for x, note in zip((60, 560, 1060), (throughput_note, wall_note, memory_note)):
        parts.extend(
            [
                rect(x + 30, 743, 420, 31, "#EEF4FF", radius=8),
                text(x + 240, 764, note, 13, fill="#29415F", weight=650, anchor="middle"),
            ]
        )
    parts.extend(
        [
            line(60, 822, 1540, 822, stroke="#CDD5DF", width=1.2),
            text(60, 857, "Qwen2.5 0.5B student · 1.5B policies · 960 trained trajectories per cell", 15, fill="#58677C", weight=550),
            text(1540, 857, "One run per cell · completions diverge after policy updates", 14, fill="#728094", anchor="end"),
            text(1540, 882, "Source: matched OPD/MOPD benchmark, 2026-08-07", 12, fill="#8A95A5", anchor="end"),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def routing_chart(data: dict, scale: float) -> str:
    width, height = 1600, 760
    domains = [("math", "Math prompts", 388), ("code", "Code prompts", 530)]
    center = 800
    axis_half = 520
    maximum = 0.6
    parts = svg_open(width, height, scale, "MOPD specialist routing affinity probe")
    parts.extend(
        [
            rect(58, 30, 184, 32, "#E5EDFF", radius=16),
            text(150, 52, "ROUTING PROBE", 13, fill="#2457B8", weight=700, anchor="middle", spacing=1.4),
            text(60, 116, "Routing worked. Specialization was asymmetric.", 45, weight=750),
            text(60, 158, "Both policies score the exact same 64 student rollouts", 22, fill="#58677C", weight=450),
            line(60, 185, 1540, 185, stroke="#CDD5DF", width=1.3),
            rect(60, 215, 1480, 430, "#FFFFFF", radius=20, stroke="#E1E6ED"),
            text(92, 262, "DOMAIN-POLICY AFFINITY", 13, fill=MUTED, weight=700, spacing=1.0),
            text(92, 300, "Mean routed-policy log-prob advantage", 23, weight=700),
            text(92, 328, "nats / response token · prompt-cluster bootstrap 95% interval", 14, fill="#718096"),
            line(center, 355, center, 590, stroke="#8391A3", width=1.5),
            text(center - axis_half, 615, "−0.6", 13, fill="#7A8798", anchor="middle"),
            text(center, 615, "0", 13, fill="#7A8798", anchor="middle"),
            text(center + axis_half, 615, "+0.6", 13, fill="#7A8798", anchor="middle"),
            text(center - 185, 365, "other policy assigns higher likelihood", 13, fill="#9A5960", anchor="middle"),
            text(center + 205, 365, "routed policy assigns higher likelihood", 13, fill="#35715C", anchor="middle"),
        ]
    )
    for domain, label, y in domains:
        row = data["domains"][domain]
        value = float(row["mean_trajectory_specialist_advantage"])
        low, high = map(float, row["prompt_cluster_bootstrap_95pct_ci"])
        value_x = center + value / maximum * axis_half
        low_x = center + low / maximum * axis_half
        high_x = center + high / maximum * axis_half
        color = "#D65D68" if value < 0 else "#27936F"
        bar_x = min(center, value_x)
        parts.extend(
            [
                text(92, y + 19, label, 20, weight=700),
                text(92, y + 45, f"routed policy wins {row['specialist_win_rate']:.0%}", 14, fill=MUTED),
                rect(bar_x, y, abs(value_x - center), 42, color, radius=7),
                line(low_x, y + 21, high_x, y + 21, stroke=INK, width=2),
                line(low_x, y + 13, low_x, y + 29, stroke=INK, width=2),
                line(high_x, y + 13, high_x, y + 29, stroke=INK, width=2),
                f'<circle cx="{value_x:.1f}" cy="{y + 21:.1f}" r="5" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>',
                text(
                    value_x + (14 if value >= 0 else -14),
                    y + 16,
                    f"{value:+.3f}",
                    16,
                    fill=color,
                    weight=750,
                    anchor="start" if value >= 0 else "end",
                ),
            ]
        )
    parts.extend(
        [
            text(60, 689, "8 prompts per domain × 4 rollouts · Qwen2.5-Math-1.5B and Qwen2.5-Coder-1.5B", 15, fill="#58677C", weight=550),
            text(1540, 689, "Likelihood affinity is not task accuracy", 14, fill="#728094", anchor="end"),
            text(1540, 724, "Source: fixed-rollout cross-score probe, 2026-08-07", 12, fill="#8A95A5", anchor="end"),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=root / "results-opd-vs-mopd.json")
    parser.add_argument("--routing", type=Path, default=root / "results-routing-probe" / "summary.json")
    parser.add_argument("--output-dir", type=Path, default=root / "charts")
    parser.add_argument("--scale", type=float, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    routing = json.loads(args.routing.read_text(encoding="utf-8"))
    outputs = {
        args.output_dir / "opd-vs-mopd-ablation.svg": ablation_chart(comparison, args.scale),
        args.output_dir / "routing-affinity-probe.svg": routing_chart(routing, args.scale),
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
