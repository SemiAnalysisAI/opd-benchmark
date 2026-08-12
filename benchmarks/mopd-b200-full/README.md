# 8×B200 full-size MOPD benchmark

This directory is the reproducible artifact for benchmarking Prime-RL, Miles,
and verl with the full model topology from Miles' Qwen3-8B multi-teacher
on-policy distillation example.

The measured workload is 15 steps of 16 prompts × 4 rollouts, using a Qwen3-8B
student, Qwen3-32B math teacher, Qwen3-Coder-30B-A3B code teacher, and a 16K
response cap on one 8×NVIDIA B200 node. Every framework receives the same data,
model revisions, optimizer settings, sampled-token reverse-KL objective, and
physical teacher placement. Framework-native scheduling differences remain
visible.

Start with:

- `REPORT.md` for measured results;
- `REPLICATION.md` for clean-node setup, execution, validation, cleanup, and
  analysis commands;
- `SOFTWARE_EXPERIENCE.md` for every failed attempt and workaround;
- `results/summary.json` for the machine-readable reduction;
- `results/all-runs/` for successful and failed raw compact logs; and
- `charts/mopd-b200-framework-benchmark.{svg,png}` for the blog chart.

Core scripts:

| File | Purpose |
|---|---|
| `setup_remote_full.sh` | Rebuild sources, packages, models, data, and containers |
| `run_all_full.sh` | Run Prime-RL, Miles, and verl serially |
| `run_*_full.sh` | Framework-specific launchers |
| `collect_environment.sh` | Capture exact host, package, image, Git, and hash manifests |
| `collect_remote_results.sh` | Download compact evidence while excluding weights |
| `analyze_full_results.py` | Validate 15 steps/framework and recompute all result metrics |
| `generate_blog_chart.py` | Regenerate SVG/PNG from `summary.json` |
| `generate_step_throughput_charts.py` | Regenerate per-step throughput SVG/PNG charts from native logs |

No base-model weights, converted weights, checkpoints, broadcast weights, or
model caches are included in the downloadable bundle.
