#!/usr/bin/env python3
"""Generate the publication SVG and PNG from measured summary.json."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path


WIDTH = 1800
HEIGHT = 1040
FONT = "Inter, Arial, Helvetica, sans-serif"
ORDER = ["prime-rl", "miles", "verl"]
LABEL = {"prime-rl": "Prime-RL", "miles": "Miles", "verl": "verl"}
STACK = {
    "prime-rl": "vLLM + native trainer",
    "miles": "SGLang + Megatron",
    "verl": "vLLM + FSDP",
}
COLOR = {"prime-rl": "#E8752B", "miles": "#7759D9", "verl": "#2875E2"}


def txt(x, y, value, size, *, fill="#10233F", weight=400, anchor="start", spacing=0):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'letter-spacing="{spacing}">{html.escape(str(value))}</text>'
    )


def rect(x, y, width, height, fill, *, radius=0, stroke=None, stroke_width=1):
    border = f' stroke="{stroke}" stroke-width="{stroke_width}"' if stroke else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{radius:.1f}" fill="{fill}"{border}/>'
    )


def line(x1, y1, x2, y2, *, stroke="#DDE3EB", width=1, dash=None):
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dashed}/>'
    )


def nice_max(value: float) -> float:
    if value <= 3:
        return 3
    if value <= 5:
        return 5
    if value <= 10:
        return 10
    if value <= 20:
        return 20
    if value <= 50:
        return 50
    if value <= 100:
        return 100
    if value <= 200:
        return 200
    if value <= 500:
        return 500
    if value <= 1000:
        return 1000
    return ((int(value) // 500) + 1) * 500


def timing(summary: dict, name: str) -> tuple[float, float]:
    row = summary["frameworks"][name]
    components = row["components"]
    if name == "prime-rl":
        generation = row["pipeline"]["mean_step_seconds"]
        trainer = components["mean_forward_backward_seconds"] + components["mean_weight_broadcast_seconds"]
    elif name == "miles":
        generation = components["mean_generation_seconds"]
        trainer = (
            components["mean_teacher_logprob_seconds"]
            + components["mean_student_logprob_seconds"]
            + components["mean_actor_train_seconds"]
            + components["mean_weight_update_seconds"]
        )
    else:
        generation = components["mean_generation_seconds"]
        trainer = (
            components["mean_old_logprob_seconds"]
            + components["mean_actor_update_seconds"]
            + components["mean_weight_update_seconds"]
        )
    return generation, trainer


def build(summary: dict) -> str:
    frameworks = summary["frameworks"]
    throughput = {name: frameworks[name]["pipeline"]["trajectories_per_second"] for name in ORDER}
    token_rate = {name: frameworks[name]["pipeline"]["tokens_per_second"] for name in ORDER}
    total_wall = {name: frameworks[name]["wall"]["wall_seconds"] for name in ORDER}
    framework_wall = {name: frameworks[name]["wall"]["framework_wall_seconds"] for name in ORDER}
    setup_wall = {name: total_wall[name] - framework_wall[name] for name in ORDER}
    stage = {name: timing(summary, name) for name in ORDER}
    best = max(throughput, key=throughput.get)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img">',
        '<title>Full-size MOPD systems benchmark on eight NVIDIA B200 GPUs</title>',
        rect(0, 0, WIDTH, HEIGHT, "#F4F6F9"),
        rect(66, 34, 214, 34, "#E4EDFF", radius=17),
        txt(173, 57, "SYSTEMS BENCHMARK", 13, fill="#2356AD", weight=750, anchor="middle", spacing=1.3),
        txt(66, 130, "MOPD throughput hinges on pipeline balance", 47, weight=780),
        txt(66, 174, "Same Qwen3-8B recipe across Prime-RL, Miles, and verl · 8× NVIDIA B200", 22, fill="#5A687B", weight=480),
        txt(1734, 58, "AUGUST 2026", 13, fill="#687587", weight=700, anchor="end", spacing=1.1),
        line(66, 205, 1734, 205, stroke="#CAD2DD", width=1.4),
        rect(66, 238, 530, 650, "#FFFFFF", radius=20, stroke="#DFE5ED"),
        rect(635, 238, 530, 650, "#FFFFFF", radius=20, stroke="#DFE5ED"),
        rect(1204, 238, 530, 650, "#FFFFFF", radius=20, stroke="#DFE5ED"),
        txt(96, 285, "REALIZED PIPELINE RATE", 14, fill="#68778A", weight=750, spacing=1.0),
        txt(96, 322, "Trajectories per second", 24, weight=720),
        txt(96, 350, "Higher is better · native step windows", 15, fill="#718095"),
        txt(665, 285, "PRODUCER / CONSUMER BALANCE", 14, fill="#68778A", weight=750, spacing=1.0),
        txt(665, 322, "Mean seconds per step", 24, weight=720),
        txt(665, 350, "Native timers · async Prime stages overlap", 15, fill="#718095"),
        txt(1234, 285, "END-TO-END WALL", 14, fill="#68778A", weight=750, spacing=1.0),
        txt(1234, 322, "Cold start + framework", 24, weight=720),
        txt(1234, 350, "Lower is better · seconds", 15, fill="#718095"),
    ]

    rows = {"prime-rl": 404, "miles": 550, "verl": 696}

    # Panel 1: throughput.
    max_tp = nice_max(max(throughput.values()) * 1.12)
    bar_x, bar_w = 235, 300
    for tick in range(5):
        value = max_tp * tick / 4
        x = bar_x + bar_w * tick / 4
        parts.extend([line(x, 385, x, 790, dash="4 7"), txt(x, 820, f"{value:g}", 13, fill="#7A8798", anchor="middle")])
    for name in ORDER:
        y = rows[name]
        width = throughput[name] / max_tp * bar_w
        parts.extend(
            [
                txt(96, y + 21, LABEL[name], 18, weight=730),
                txt(96, y + 43, STACK[name], 11, fill="#7B8797"),
                rect(bar_x, y, bar_w, 49, "#EDF1F5", radius=7),
                rect(bar_x, y, width, 49, COLOR[name], radius=7),
                txt(bar_x + width + 10, y + 31, f"{throughput[name]:.2f}", 17, weight=760),
                txt(bar_x, y + 72, f"{token_rate[name]:,.0f} trained tok/s", 12, fill="#69788C"),
            ]
        )

    # Panel 2: generation cadence versus trainer/update path.
    max_stage = nice_max(max(max(values) for values in stage.values()) * 1.12)
    center_x, center_w = 815, 286
    for tick in range(5):
        value = max_stage * tick / 4
        x = center_x + center_w * tick / 4
        parts.extend([line(x, 385, x, 790, dash="4 7"), txt(x, 820, f"{value:g}", 13, fill="#7A8798", anchor="middle")])
    parts.extend(
        [
            rect(872, 365, 14, 14, "#2D6AA8", radius=3),
            txt(894, 377, "generation cadence", 12, fill="#647388"),
            rect(1010, 365, 14, 14, "#9DB8D2", radius=3),
            txt(1032, 377, "trainer + sync", 12, fill="#647388"),
        ]
    )
    for name in ORDER:
        y = rows[name]
        gen, trainer = stage[name]
        parts.extend(
            [
                txt(665, y + 29, LABEL[name], 18, weight=730),
                rect(center_x, y, gen / max_stage * center_w, 20, "#2D6AA8", radius=5),
                rect(center_x, y + 29, trainer / max_stage * center_w, 20, "#9DB8D2", radius=5),
                txt(center_x + gen / max_stage * center_w + 8, y + 16, f"{gen:.1f}s", 12, weight=680),
                txt(center_x + trainer / max_stage * center_w + 8, y + 45, f"{trainer:.1f}s", 12, weight=680),
            ]
        )

    # Panel 3: setup + measured framework wall.
    max_wall = nice_max(max(total_wall.values()) * 1.08)
    wall_x, wall_w = 1385, 285
    for tick in range(5):
        value = max_wall * tick / 4
        x = wall_x + wall_w * tick / 4
        parts.extend([line(x, 385, x, 790, dash="4 7"), txt(x, 820, f"{value:g}", 13, fill="#7A8798", anchor="middle")])
    parts.extend(
        [
            rect(1428, 365, 14, 14, "#D5DDE7", radius=3),
            txt(1450, 377, "teacher/startup", 12, fill="#647388"),
            rect(1571, 365, 14, 14, "#416F9B", radius=3),
            txt(1593, 377, "framework", 12, fill="#647388"),
        ]
    )
    for name in ORDER:
        y = rows[name]
        setup_w = setup_wall[name] / max_wall * wall_w
        run_w = framework_wall[name] / max_wall * wall_w
        parts.extend(
            [
                txt(1234, y + 29, LABEL[name], 18, weight=730),
                rect(wall_x, y, setup_w, 49, "#D5DDE7", radius=7),
                rect(wall_x + setup_w, y, run_w, 49, COLOR[name], radius=7),
                txt(wall_x + setup_w + run_w + 8, y + 31, f"{total_wall[name]:.0f}s", 16, weight=760),
            ]
        )

    parts.extend(
        [
            rect(96, 848, 470, 26, "#EFF4FB", radius=8),
            txt(112, 866, f"Highest realized rate: {LABEL[best]} · {throughput[best]:.2f} traj/s", 13, fill="#294969", weight=700),
            rect(665, 848, 470, 26, "#EFF4FB", radius=8),
            txt(681, 866, "Generation dominates Miles/verl; Prime nearly matches stages", 13, fill="#294969", weight=700),
            rect(1234, 848, 470, 26, "#EFF4FB", radius=8),
            txt(1250, 866, "Wall includes cold teacher and framework initialization", 13, fill="#294969", weight=700),
            line(66, 926, 1734, 926, stroke="#CAD2DD", width=1.3),
            txt(66, 962, "Qwen3-8B student · Qwen3-32B math + Qwen3-Coder-30B-A3B teachers · 15 steps · 960 real trajectories", 15, fill="#56667A", weight=570),
            txt(66, 992, "Sampled-token reverse KL · 16K response cap · one run per framework · no model-quality claim", 14, fill="#758296"),
            txt(1734, 962, "8× B200 · source: reproducible benchmark logs", 14, fill="#647388", anchor="end", weight=620),
            txt(1734, 992, "SemiAnalysis OPD systems benchmark · August 2026", 12, fill="#8994A3", anchor="end"),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=root / "results" / "summary.json")
    parser.add_argument("--svg", type=Path, default=root / "charts" / "mopd-b200-framework-benchmark.svg")
    parser.add_argument("--png", type=Path, default=root / "charts" / "mopd-b200-framework-benchmark.png")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rendered = build(summary)
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(rendered, encoding="utf-8")
    try:
        import cairosvg
    except (ImportError, OSError):
        sips = shutil.which("sips")
        if not sips:
            print(f"wrote {args.svg}; install CairoSVG to emit PNG")
            return
        subprocess.run(
            [sips, "-s", "format", "png", str(args.svg), "--out", str(args.png)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    else:
        cairosvg.svg2png(
            bytestring=rendered.encode(),
            write_to=str(args.png),
            output_width=WIDTH,
            output_height=HEIGHT,
        )
    print(args.svg)
    print(args.png)


if __name__ == "__main__":
    main()
