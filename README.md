# On-policy distillation

On-policy distillation (OPD) experiments across training frameworks.
The current experiment is multi-teacher OPD (MOPD). Qwen3.6-35B-A3B learns from frozen reasoning_gym GRPO teachers for `caesar_cipher` and `simple_geometry`.
**Every experiment runs with thinking on** (32k context, up to 30,720 response tokens), in every framework, for training, teachers, and evaluation.

**This repository's code has run the single-teacher `caesar_cipher` variant (`--domains caesar_cipher`) in Miles, Prime-RL, Slime and verl on GPUs; NeMo-RL deadlocked before its first update. See the [benchmark report](docs/CAESAR-OPD-2026-09-27.md). It has not run the two-teacher MOPD experiment.** A separate research campaign ran the two-teacher experiment.
A previous experiment (Countdown and Graph Coloring, thinking off) completed with Miles, Prime-RL, and Slime on two nodes of eight B200 GPUs.
The runtime image and converted checkpoints are external prerequisites.

## Start here

To rerun the whole caesar_cipher benchmark (every framework, final-model benchmarks, telemetry report) with the production setup,
fill in `config/site.local.json` and run `PARTITION=YOUR_PARTITION tools/reproduce.sh all` on a Slurm login node; see [setup](docs/SETUP.md#one-command-toolsreproducesh).
Step by step:

1. Read [the framework overview](frameworks/README.md) and [the setup guide](docs/SETUP.md).
2. Copy `config/site.example.json` to `config/site.local.json` and fill in your nodes, addresses, and paths.
3. `python3 tools/prepare.py slime --site config/site.local.json --output /shared/opd-runs/slime-01`
4. `python3 tools/submit.py /shared/opd-runs/slime-01 --partition YOUR_PARTITION` prints the allocation. Add `--submit` to allocate 16 GPUs.

Use `miles`, `prime-rl`, `verl`, or `nemo-rl` (three nodes, `config/site-nemo-rl.example.json`) for the other frameworks. Preparation never overwrites a directory, submits a job, or downloads weights.

## Contents

| Path | Contents |
|---|---|
| `shared/` | Shared recipe, data checks, verifier, Slurm and container runtime, telemetry. |
| `frameworks/` | Miles, NeMo-RL, Prime-RL, Slime, and verl code, each with its pinned `upstream/` revision (and patch, where one is needed). |
| `hosted/` | Tinker and Fireworks campaigns with their own protocols. |
| `data/` | The exact compressed task rows and their hashes. |
| `config/` | Example site and hosted-account files. See [its README](config/README.md). |
| `tools/` | Preparation, teacher fusing, guarded submission, source verification, benchmarking, reporting, and `reproduce.sh` (the whole benchmark end to end). |
| `docs/` | [Setup](docs/SETUP.md), [provenance](docs/PROVENANCE.md), [validation](docs/VALIDATION.md), and the [caesar_cipher OPD benchmark](docs/CAESAR-OPD-2026-09-27.md) of all five frameworks. |
| `article/` | [Article viewer, drafts, figures, charts, and editorial research](article/README.md). |

The implementations differ in objective, scheduling, precision, and checkpoint content.
Results are single observations and do not rank the frameworks. Maximum useful GPU utilization was not demonstrated.

## Checks

Python 3.12 is required. These checks need no GPU.

```bash
python3 -m unittest discover -s tests -v
git diff --check
```

Raw logs, weights, checkpoints, credentials, and local config files are excluded from Git.
Earlier repository contents remain in Git history at `a1bf83efd1a5cf11ef72b3ea18eb98132d83c92e`.
