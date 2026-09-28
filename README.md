# On-policy distillation

On-policy distillation (OPD) experiments across five training frameworks: Miles, NeMo-RL, Prime-RL, Slime and verl.

The current experiment is multi-teacher OPD (MOPD): Qwen3.6-35B-A3B learns from frozen reasoning_gym GRPO teachers for `caesar_cipher` and `simple_geometry`. Thinking is on everywhere (training, teachers and evaluation), with a 32k context and up to 30,720 response tokens.

## Status

- The single-teacher `caesar_cipher` variant (`--domains caesar_cipher`) has run on GPUs in Miles, Prime-RL, Slime and verl. NeMo-RL deadlocked before its first update. See the [benchmark report](docs/CAESAR-OPD-2026-09-27.md).
- The two-teacher MOPD experiment has not been run with this code.
- A previous experiment (Countdown and Graph Coloring, thinking off) completed in Miles, Prime-RL and Slime on two nodes of eight B200 GPUs. See [provenance](docs/PROVENANCE.md).

Results are single observations and do not rank the frameworks. The implementations differ in objective, scheduling, precision and checkpoint content, and maximum useful GPU utilization was not demonstrated.

## Quick start

The runtime image and converted checkpoints are external prerequisites; the [setup guide](docs/SETUP.md) covers them.

To rerun the whole `caesar_cipher` benchmark (every framework, checkpoint benchmarks and the telemetry report), fill in `config/site.local.json` and run this on a Slurm login node:

```bash
PARTITION=YOUR_PARTITION tools/reproduce.sh all
```

See [One command: `tools/reproduce.sh`](docs/SETUP.md#one-command-toolsreproducesh) for its settings.

To run one framework by hand:

1. Read the [framework overview](frameworks/README.md) and the [setup guide](docs/SETUP.md).
2. Copy `config/site.example.json` to `config/site.local.json` and fill in your nodes, addresses and paths.
3. Prepare a campaign:
   ```bash
   python3 tools/prepare.py slime --site config/site.local.json --output /shared/opd-runs/slime-01
   ```
4. Print the allocation, then add `--submit` to allocate 16 GPUs:
   ```bash
   python3 tools/submit.py /shared/opd-runs/slime-01 --partition YOUR_PARTITION
   ```

Replace `slime` with `miles`, `prime-rl`, `verl` or `nemo-rl`. NeMo-RL needs three nodes and `config/site-nemo-rl.example.json`. Preparation never overwrites a directory, submits a job or downloads weights.

## Layout

| Path | Contents |
|---|---|
| `shared/` | Recipe, data checks, verifier, Slurm and container runtime, telemetry |
| `frameworks/` | Per-framework code, each with its pinned `upstream/` revision and patch, if any |
| `hosted/` | Tinker and Fireworks campaigns, each with its own protocol, and the hosted OPD/MOPD loop ([README](hosted/opd/README.md)) |
| `data/` | Compressed task rows and their hashes |
| `config/` | Example site and hosted-account files ([README](config/README.md)) |
| `tools/` | Preparation, teacher fusing, guarded submission, source verification, benchmarking, reporting, `reproduce.sh` |
| `docs/` | [Setup](docs/SETUP.md), [provenance](docs/PROVENANCE.md), [validation](docs/VALIDATION.md), [caesar_cipher benchmark](docs/CAESAR-OPD-2026-09-27.md) |

## Checks

Python 3.12 is required. These checks need no GPU:

```bash
python3 -m unittest discover -s tests -v
git diff --check
```

Raw logs, weights, checkpoints, credentials and local config files are kept out of Git. Earlier repository contents remain in history at `a1bf83efd1a5cf11ef72b3ea18eb98132d83c92e`.
