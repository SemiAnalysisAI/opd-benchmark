# Reproducing the NeMo-RL v0.7.0 problems

Everything here uses the benchmark's production setup (Qwen3.6-35B-A3B, the caesar_cipher teacher,
the shared recipe) or no model at all. The full log of our attempts is in
[`docs/caesar-opd-2026-09-27/nemo-rl-log.md`](../../../docs/caesar-opd-2026-09-27/nemo-rl-log.md).

## The production run

```bash
PARTITION=your-partition NEMO_RL_SITE=config/site-nemo-rl.local.json tools/reproduce.sh nemo-rl
```

This downloads the pinned models, fuses the teacher, imports `nvcr.io/nvidia/nemo-rl:v0.7.0`, prepares
the campaign with `tools/prepare.py nemo-rl --domains caesar_cipher` and submits it: MOPD on async GRPO
with NeMo Gym rollouts, launched with the container's `ray.sub` on three 8-GPU nodes. The NeMo-RL site
file is the two-node one plus `teacher_node`, `teacher_ip` and `nemo_rl_container`.

If the first trainer-to-vLLM weight sync has not finished after `DEADLOCK_MIN` minutes (default 20), a
watchdog cancels the job, prints the vLLM workers' weight-update errors and writes them to the run's
`DEADLOCK` file. Without it, the deadlocked job holds 24 GPUs until its time limit.

## Three container checks (one GPU, about two minutes each)

These need Slurm with Pyxis/Enroot and network access to NGC, and nothing from this repository
except `repro.sh`.

```bash
export PARTITION=your-partition WORKDIR=/path/to/nemo-rl-repro
# Optional: export CONTAINER=/path/to/nemo-rl-v0.7.0.sqsh  (enroot import of the image, 43 GB)
./repro.sh image
./repro.sh uv-override
./repro.sh readonly
```

Results on 8 x B200 nodes (Slurm on Kubernetes, Pyxis, Lustre), 28 September 2026:

| Check | What it runs | Result |
|---|---|---|
| `image` | Reads the container's own metadata | `NEMO_RL_COMMIT=19244a09`, not the `v0.7.0` tag's `81aa43dd` (the tag only adds docs and a version bump). The venv's files are symlinks into the image's uv cache (`transformers/__init__.py -> /root/.cache/uv/archive-v0/...`). All three Qwen3.5-35B-A3B recipes set `enforce_eager: true`; the base config default is `False`. |
| `uv-override` | Mounts a directory over `/root/.cache/uv`, as `ray.sub`'s documented `UV_CACHE_DIR_OVERRIDE` does, then runs the first import in `nemo_rl/algorithms/grpo.py` | `ImportError: cannot import name 'AutoProcessor' from 'transformers' (unknown location)`: hiding the cache turns every venv package into a dangling symlink. |
| `readonly` | Runs upstream's documented `uv run` in the container the way `ray.sub` starts it (no `--container-writable`, so read-only by Pyxis' default) | `uv` fails with `Read-only file system` at `/root/.cache/uv`. NeMo-RL also patches vLLM's installed files at run time (`nemo_rl/models/generation/vllm/patches.py`), so the launcher only works with writable containers. The package sets `PYXIS_CONTAINER_WRITABLE=1`. |

## Why a failed weight update becomes a silent deadlock

1. vLLM's weight loader rejects the fused MoE expert tensors NeMo-RL sends for Qwen3.6-35B-A3B:
   `shard_dim=0 is not a valid data dimension for a 3D tensor (expected 1 or 2)`. Every vLLM worker
   logged this in job 1010 (Megatron EP8).
2. NeMo-RL catches the exception on the vLLM side, prints it and returns `False`
   (`nemo_rl/models/generation/vllm/vllm_backend.py`, `update_weights_from_collective`, lines 466-470).
3. The driver waits for the trainer's broadcast before it checks vLLM's result
   (`nemo_rl/algorithms/grpo.py` lines 1946-1947: `ray.get(futures_train)`, then
   `ray.get(futures_inference)`). The receivers have already left the collective, so the broadcast
   never completes and the check that would raise never runs. The job holds its GPUs until its time
   limit.

Ray prints the worker error with a `[repeated Nx across cluster]` suffix, which log scans easily miss;
the `tools/reproduce.sh` watchdog prints it explicitly.

## Our earlier runs

Three clean starts of our configuration (jobs 1001-1003) got through setup and the pre-training
validation, then stopped in the first weight sync, before the first update (`async_grpo_train` →
`refit_policy_generation`, `grpo.py:3553`), with every GPU idle. This is the same call that deadlocked
in job 1010. Rollouts for the first batch still finished, because NeMo-RL collects them in the background
during that sync.

py-spy, run from the host (Pyxis containers share the host's process table), showed the trainer ranks
waiting in `stream.synchronize()` inside `packed_broadcast_producer` (`nemo_rl/utils/packed_tensor.py:68`)
and the vLLM engines idle, which is also what engines that had already failed and returned would look
like. The driver output of those runs was buffered (no `PYTHONUNBUFFERED`) and shows no refit lines or
worker errors, so we cannot confirm they hit the `shard_dim` error. All three had vLLM CUDA graphs on,
and the hang was the same with eight single-GPU (TP1) engines. The package now runs the driver
unbuffered, so `tools/reproduce.sh nemo-rl` records the errors.
