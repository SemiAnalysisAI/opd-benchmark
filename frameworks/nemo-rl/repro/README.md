# Reproducing the NeMo-RL v0.7.0 problems

Everything here uses the production setup of the benchmark (Qwen3.6-35B-A3B, the caesar_cipher teacher,
the shared recipe) or no model at all. The full record of our attempts is in
[`docs/caesar-opd-2026-09-27/nemo-rl-log.md`](../../../docs/caesar-opd-2026-09-27/nemo-rl-log.md).

## The production run, end to end

```bash
PARTITION=your-partition NEMO_RL_SITE=config/site-nemo-rl.local.json tools/reproduce.sh nemo-rl
```

This downloads the pinned models, fuses the teacher, imports `nvcr.io/nvidia/nemo-rl:v0.7.0`, prepares
the campaign with `tools/prepare.py nemo-rl --domains caesar_cipher` and submits it: MOPD on async GRPO
with NeMo Gym rollouts, launched with the `ray.sub` shipped in the container, on three 8-GPU nodes. A
watchdog cancels the run if its first trainer-to-vLLM weight sync has not completed after
`DEADLOCK_MIN` minutes (default 20), prints the vLLM workers' weight-update errors, and writes them to
the run's `DEADLOCK` file; otherwise the deadlocked job holds 24 GPUs until its time limit. The NeMo-RL
site file is the two-node one plus `teacher_node`, `teacher_ip` and `nemo_rl_container`.

## Three container checks (one GPU, about two minutes each)

Needs Slurm with Pyxis/Enroot and network access to NGC; nothing from this repository but `repro.sh`.

```bash
export PARTITION=your-partition WORKDIR=/shared/you/nemo-rl-repro
# Optional: export CONTAINER=/shared/you/nemo-rl-v0.7.0.sqsh  (enroot import of the image, 43 GB)
./repro.sh image
./repro.sh uv-override
./repro.sh readonly
```

What we saw on 8 x B200 nodes (Slurm on Kubernetes, Pyxis, Lustre) on 28 September 2026:

| Check | What it runs | What we saw |
|---|---|---|
| `image` | Reads the container's own metadata | `NEMO_RL_COMMIT=19244a09`, not the `v0.7.0` tag's `81aa43dd` (the tag adds only docs and a version bump). The venv's files are symlinks into the image's uv cache (`transformers/__init__.py -> /root/.cache/uv/archive-v0/...`). All three Qwen3.5-35B-A3B recipes set `enforce_eager: true`; the base config default is `False`. |
| `uv-override` | Mounts a directory over `/root/.cache/uv`, exactly as `ray.sub`'s documented `UV_CACHE_DIR_OVERRIDE` does, then runs the first import in `nemo_rl/algorithms/grpo.py` | `ImportError: cannot import name 'AutoProcessor' from 'transformers' (unknown location)`: hiding the cache leaves every venv package a dangling symlink. |
| `readonly` | Runs upstream's documented `uv run` in the container as `ray.sub` starts it (no `--container-writable`, which is Pyxis' default) | `uv` fails: `Read-only file system` at `/root/.cache/uv`. At run time NeMo-RL also patches vLLM's installed files in place (`nemo_rl/models/generation/vllm/patches.py`), so the launcher only works where containers are writable. The package sets `PYXIS_CONTAINER_WRITABLE=1`. |

## Why a failed weight update becomes a silent deadlock

1. vLLM's weight loader rejects the fused MoE expert tensors NeMo-RL sends for Qwen3.6-35B-A3B
   (`shard_dim=0 is not a valid data dimension for a 3D tensor (expected 1 or 2)`; logged by every vLLM
   worker in job 1010, Qwen3.6-35B-A3B with Megatron EP8).
2. NeMo-RL catches that exception on the vLLM side, prints it, and returns `False`
   (`nemo_rl/models/generation/vllm/vllm_backend.py`, `update_weights_from_collective`, lines 466-470).
3. The driver waits for the trainer's broadcast before it looks at vLLM's result
   (`nemo_rl/algorithms/grpo.py` lines 1946-1947: `ray.get(futures_train)`, then
   `ray.get(futures_inference)`). The receivers have left the collective, so the trainer's NCCL broadcast
   never completes and the success check that would raise never runs. The job holds its GPUs until
   its time limit.

Ray prints the worker error with a `[repeated Nx across cluster]` suffix, which is easy to filter out
of a log scan; the `tools/reproduce.sh` watchdog prints it.

## Our own runs

In three clean starts of our own configuration (jobs 1001-1003) the run got through set-up and the
pre-training validation, then stopped in the first weight sync, the one before the first update
(`async_grpo_train` → `refit_policy_generation`, `grpo.py:3553`), with every GPU idle. Rollouts for the
first batch still finished, because NeMo-RL collects them in the background during that sync. This is the
same call that deadlocked in job 1010. py-spy (from the host; Pyxis containers share the host's process
table) showed the trainer ranks in `packed_broadcast_producer` (`nemo_rl/utils/packed_tensor.py:68`) waiting
in `stream.synchronize()` and the vLLM engines idle, which is also what engines that had already failed and
returned would look like. Those runs' driver output was buffered (no `PYTHONUNBUFFERED`) and shows none of
the refit lines or worker errors, so we cannot confirm they hit the `shard_dim` error. It hung the same way
with eight single-GPU (TP1) engines, and all three had vLLM CUDA graphs on. The package now runs the driver
unbuffered, so `tools/reproduce.sh nemo-rl` records the errors.
