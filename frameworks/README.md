# Frameworks

Each directory runs the experiment defined in `shared/recipe.py` on Slurm: a Qwen3.6-35B-A3B student learns from frozen reasoning_gym GRPO teachers, one for `caesar_cipher` and one for `simple_geometry`. A directory holds only framework-specific code. `upstream/` holds the pinned revision, the patch (if any) and the license.

The single-teacher `caesar_cipher` variant (`--domains caesar_cipher`) has run in Miles, Prime-RL, Slime and verl; NeMo-RL deadlocked before its first update. See the [benchmark report](../docs/CAESAR-OPD-2026-09-27.md). The two-teacher experiment has not been run with this code. See [setup](../docs/SETUP.md) to run it, and [provenance](../docs/PROVENANCE.md) for the previous experiment's results.

## Shared recipe (`shared/recipe.py`)

| Setting | Value |
|---|---|
| Tasks | reasoning_gym 0.1.25 `caesar_cipher`, `simple_geometry` |
| Updates | 20 |
| Prompts per update | 128, split equally between tasks, one sample each |
| Learning rate | 1e-6 |
| Thinking | On |
| Context / response | 32,768 / up to 30,720 tokens |
| Maximum policy lag | 1 update |
| Checkpoints | After updates 10 and 20 (`tools/reproduce.sh` benchmarks both) |
| Evaluation | Every 10 updates, greedy, on the full dev sets (126 / 128) |
| Megatron trainer memory | No context parallelism; 32,768 max tokens per GPU; log-prob chunks of 4096 |
| Trainer node | 8 GPUs |
| Generation node | Teachers on GPUs 0-1; policy on GPUs 2-7 in 2-GPU engines |
| CPUs | 80 per node; 64 exposed to Ray |

`shared/scoring.py` scores the text after the last `</think>` with reasoning_gym's `score_answer`. The score is diagnostic only.

## Implementations

| | Miles | Prime-RL | Slime | verl |
|---|---|---|---|---|
| Objective | Sampled-token OPD (`--loss-mode legacy --candidate-top-k 0`) | Native per-source OPD through reference KL | Native OPD through an advantage correction | Sampled-token k1 reverse KL as a policy gradient (`distillation`, no task reward) |
| Scheduling | Fully async; domain-balanced source and buffer | Continuous async | `train_async`, one rollout ahead | Fully async trainer, staleness 1, weight sync after every update |
| Trainer | Megatron | Prime-RL trainer (FSDP) | Megatron | FSDP |
| Policy serving | SGLang through Ray | Three TP2 vLLM engines behind vllm-router | SGLang through Ray | Three TP2 vLLM engines in verl's rollout pool |
| Teacher serving | Prefill-only SGLang (`shared/container.py`) | Single-GPU vLLM per teacher | Prefill-only SGLang (`shared/container.py`) | Single-GPU vLLM per teacher in verl's teacher pool, routed by `data_source` |
| Runtime | Pyxis container | Host virtual environment | Pyxis container | Host virtual environment (verl's `uv.lock`) |
| Recipe in code | `MilesNode.train_command` in `node.py` | `recipe_input()` in `make_config.py` | `train.py` | `overrides()` in `train.py` |

### Miles

Miles uses sampled tokens because top-k candidate scoring is quadratic in response length. Its launcher arguments also disable the answer stop, set a 7200 s teacher timeout, match Slime's Adam settings (weight decay 0, beta2 0.999), drop the MTP layers and replace the puzzle-only data source. The learning rate is the pinned launcher's default.

### Prime-RL

Prime-RL keeps upstream defaults. Only the shared recipe values (including policy lag 1 and weight decay 0), the two-node layout and the teachers are set. The teacher engine takes the two settings from upstream's OPD example (`gpu_memory_utilization` 0.5, eager), and the router takes upstream's arguments.

The student trains as a text model (no `model.vlm`), so the trainer keeps fp32 master weights. A one-file patch gives the frozen vision encoder SDPA, because FlashAttention 4 rejects fp32 inputs.

### verl

verl runs unmodified and follows upstream `examples/on_policy_distillation_trainer` on `verl.experimental.fully_async_policy`. It trains with FSDP and places its own trainer, rollout and teacher pools on the two-node Ray cluster, so Ray chooses the physical GPU assignment on the generation node. It streams prompts, so each update has 128 responses but not an exact 64/64 task split.

### Memory

Miles, Slime and verl set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on the trainer (verl sets it on every Ray node).

## NeMo-RL

`nemo-rl/` runs NeMo-RL's MOPD (async GRPO with `adv_estimator: opd`, rollouts through NeMo Gym's `reasoning_gym` server) unmodified from `nvcr.io/nvidia/nemo-rl:v0.7.0` (commit `19244a09`), launched with upstream `ray.sub`.

The recipe is `recipe_input()` in `make_config.py`: the shared recipe on upstream's base GRPO config, the loss settings of the MOPD reference recipe, Megatron EP8 with no context parallelism, eight single-GPU vLLM engines with CUDA graphs, and the teacher on one GPU. MOPD reserves whole nodes for the teacher and for generation, so it needs three nodes (24 GPUs) and supports only one task.

Every run so far deadlocked in the first weight sync. [`nemo-rl/repro/`](nemo-rl/repro/README.md) documents and reproduces it.

## Hosted variants

[Tinker](../hosted/tinker/README.md) and [Fireworks](../hosted/fireworks/README.md) use hosted training APIs instead of `tools/`. Their protocols are in their READMEs.
