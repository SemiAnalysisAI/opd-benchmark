# Frameworks

Miles, NeMo-RL, Prime-RL, Slime, and verl each implement one fixed multi-teacher OPD (MOPD) experiment on Slurm.
A Qwen3.6-35B-A3B student learns from two frozen reasoning_gym GRPO teachers, one for `caesar_cipher` and one for `simple_geometry`.
Each directory holds only framework-specific code. `upstream/` holds the pinned revision, patch, and license.

## Shared recipe (`shared/recipe.py`)

| Setting | Value |
|---|---|
| Tasks | reasoning_gym 0.1.25 `caesar_cipher`, `simple_geometry` |
| Updates | 20 |
| Prompts per update | 128, half per task, one sample each |
| Learning rate | 1e-6 |
| Thinking | On |
| Context / response | 32,768 / up to 30,720 tokens |
| Maximum policy lag | 1 update |
| Checkpoint | After updates 10 and 20 (each benchmarked by `tools/reproduce.sh`) |
| Evaluation | Every 10 updates; full dev sets (126 / 128); greedy |
| Megatron trainer memory | No context parallelism; 32,768 max tokens per GPU; log-prob chunks of 4096 |
| Trainer node | 8 GPUs |
| Generation node | Teachers on GPUs 0-1; policy on GPUs 2-7 in 2-GPU engines |
| CPUs | 80 per node; 64 exposed to Ray |

`shared/scoring.py` scores the text after the last `</think>` with reasoning_gym's `score_answer`. The score is diagnostic only.
These settings follow the research campaign in `research/mopd-2026-09-25`, which had no in-loop evaluation.

## Implementations

| | Miles | Prime-RL | Slime | verl |
|---|---|---|---|---|
| Objective | Sampled-token OPD (`--loss-mode legacy --candidate-top-k 0`) | Native per-source OPD through reference KL | Native OPD through an advantage correction | Sampled-token k1 reverse KL as a policy gradient (`distillation`, no task reward) |
| Scheduling | Fully async; domain-balanced source and buffer | Continuous async | `train_async`, one rollout ahead | Fully async trainer, staleness 1, weight sync after every update |
| Policy serving | SGLang through Ray | Three TP2 vLLM engines behind vllm-router | SGLang through Ray | Three TP2 vLLM engines in verl's rollout pool |
| Teacher serving | Prefill-only SGLang (`shared/container.py`) | Single-GPU vLLM per teacher | Prefill-only SGLang (`shared/container.py`) | Single-GPU vLLM per teacher in verl's teacher pool, routed by `data_source` |
| Runtime | Pyxis container | Host virtual environment | Pyxis container | Host virtual environment (verl's `uv.lock`) |
| Recipe in code | `MilesNode.train_command` in `node.py` | `recipe_input()` in `make_config.py` | `train.py` | `overrides()` in `train.py` |

Miles uses sampled tokens because top-k candidate scoring is quadratic in response length.
Its launcher arguments also disable the answer stop, set a 7200 s teacher timeout, match Slime's Adam settings (weight decay 0, beta2 0.999),
drop the MTP layers, and replace the puzzle-only data source. The learning rate is the pinned launcher's default.
Prime-RL runs unmodified upstream with its defaults: only the shared recipe values (including policy lag 1 and weight decay 0),
the two-node layout and the teachers are set, the teacher engine takes the two settings of upstream's OPD example
(`gpu_memory_utilization` 0.5, eager), and the router takes upstream's arguments. The student trains as a text model
(no `model.vlm`), so the trainer keeps fp32 master weights; a one-file patch gives the frozen vision encoder SDPA, which
FlashAttention 4 needs for fp32 inputs.
Miles and Slime set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on the trainer node.
verl follows upstream `examples/on_policy_distillation_trainer` on `verl.experimental.fully_async_policy`, unmodified.
It trains with Megatron through mbridge (EP8, no context parallelism) and places its own trainer, rollout, and teacher pools on the two-node Ray cluster,
so its physical GPU assignment on the generation node is chosen by Ray. It streams prompts, so each update has 128 responses but not an exact 64/64 split.

**This repository's code has run the single-teacher `caesar_cipher` variant (`--domains caesar_cipher`) in Miles, Prime-RL, Slime and verl on GPUs; NeMo-RL deadlocked before its first update. See the [benchmark report](../docs/CAESAR-OPD-2026-09-27.md). It has not run the two-teacher MOPD experiment.** For the previous experiment's recorded results, see [provenance](../docs/PROVENANCE.md).
See [setup](../docs/SETUP.md) to run it.

## NeMo-RL

`nemo-rl/` runs NeMo-RL's MOPD (async GRPO with `adv_estimator: opd`, rollouts through NeMo Gym's `reasoning_gym` server)
unmodified from `nvcr.io/nvidia/nemo-rl:v0.7.0` (commit `19244a09`), launched with upstream `ray.sub`. The recipe is in
`recipe_input()` in `make_config.py`: the shared recipe on upstream's base GRPO config, the MOPD reference recipe's loss settings,
Megatron EP8 with no context parallelism, eight single-GPU vLLM engines with CUDA graphs, and the teacher on one GPU.
MOPD reserves whole nodes for the teacher and for generation, so it needs three nodes (24 GPUs) and supports one domain only.
Every run so far deadlocked in the first weight sync; [`nemo-rl/repro/`](nemo-rl/repro/README.md) documents and reproduces it.

## Hosted variants

[Tinker](../hosted/tinker/README.md) and [Fireworks](../hosted/fireworks/README.md) use hosted training APIs, not `tools/`. Their protocols are in their READMEs.
