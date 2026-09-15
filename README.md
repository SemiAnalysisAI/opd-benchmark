# Multi-teacher on-policy distillation

This repository preserves the scripts for three completed experiments with Miles, Prime-RL, and Slime.
Each experiment trains Qwen3.6-35B-A3B with frozen Countdown and Graph Coloring teachers.

**The original experiments completed. The portable package has passed local checks, but it has not been rerun on GPUs.**
The runtime image and converted model checkpoints are external prerequisites. This repository is not a self-contained container distribution.

## Start here

1. Read [the setup guide](docs/SETUP.md), including the runtime and checkpoint requirements.
2. Copy `site.example.json` to `site.local.json` and supply your node names, addresses, and paths.
3. Run `python3 tools/prepare.py slime --site site.local.json --output /shared/opd-runs/slime-01` on the shared filesystem.
4. Complete the framework setup and inspect the command with `python3 tools/submit.py /shared/opd-runs/slime-01 --partition YOUR_PARTITION`.
5. Add `--submit` only when you intend to allocate 16 GPUs. The command remains attached and writes `allocation.out`.

Replace `slime` with `miles` or `prime-rl` to prepare another framework. Use a new campaign directory for each experiment.
Preparation clones source and writes files, but does not submit a GPU job or download model weights.
An existing campaign directory is never overwritten.

## What is included

| Directory | Contents |
|---|---|
| `frameworks/miles/` | The fully async candidate-based objective, domain-balanced buffer, scoring retries, and instrumentation. |
| `frameworks/prime-rl/` | Native per-source OPD, configuration generation, task environment, and the frozen-vision compatibility patch. |
| `frameworks/slime/` | Native batched async OPD, teacher routing, baseline evaluation, timing, and runtime compatibility patches. |
| `data/` | The exact compressed puzzle files and hashes of their uncompressed contents. |
| `tools/` | Preparation, guarded Slurm submission, and source reconstruction checks. |

The files in `campaign/` are templates. Values such as `@BASE_MODEL@` are replaced during preparation.
Run the prepared copies rather than the templates. Each manifest records the upstream revision, patch digest, and original archive digest.
The [provenance document](docs/PROVENANCE.md) identifies packaging changes and scientific differences.

## Fixed experiment

The experiments use 40 optimizer updates, 128 prompts per update, one response per prompt, learning rate 1e-6, and a 256-token response limit.
The GPU allocation contains eight trainer GPUs, six policy GPUs, and one GPU per frozen teacher.
Each experiment uses two nodes with eight NVIDIA B200 GPUs each. Evaluation uses 512 development prompts per domain with thinking disabled.

The implementations have different objectives, scheduling, precision choices, and checkpoint content.
These experiments are individual observations and do not establish a framework ranking.
The sampled utilization did not demonstrate maximum useful utilization. No GPU profiler was enabled.

## Checks

Python 3.12 is required for the preparation tools and archived campaign code.
The following checks need no GPU and do not contact the cluster.

```bash
python3 -m unittest discover -s tests -v
git diff --check
```

Raw logs, model weights, generated checkpoints, local credentials, and site configuration are excluded from Git.
Keep run artifacts outside this checkout and retain their source snapshot, resolved configuration, exit status, and checksums.
The old repository contents remain in Git history at `a1bf83efd1a5cf11ef72b3ea18eb98132d83c92e`.
This change does not remove old data from Git history or change repository visibility.
