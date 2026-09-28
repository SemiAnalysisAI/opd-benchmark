# NeMo-RL attempt log

What we tried with NeMo-RL for the [caesar_cipher OPD benchmark](../CAESAR-OPD-2026-09-27.md), and why no
training update completed.

- Upstream: `NVIDIA-NeMo/RL`. We first read `main` at `4aaa48fabd178a4bf481e52c49e9125995e0f2a8`
  (2026-09-24), then ran the NGC container `nvcr.io/nvidia/nemo-rl:v0.7.0`, which is built from `19244a09`.
- Mode: MOPD on async GRPO (`adv_estimator: opd`) with NeMo Gym rollouts, per
  `docs/about/algorithms/mopd.md` and `examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack.yaml`.

## Findings before any GPU time

- **Two distillation paths.** `run_distillation.py` does teacher-logit KD and "currently supports the
  DTensor and vLLM generation backend. Megatron generation/training paths are not supported yet". MOPD
  is async GRPO with a sampled-token teacher-minus-student advantage, the same objective as the other
  frameworks. (The on-policy-distillation guide's Megatron note conflicts with a checked-in
  `distillation_math_megatron.yaml`; not investigated, since we used MOPD.)
- **Layout.** MOPD teachers are non-colocated and reserve whole nodes (`opd_teacher_nodes += num_nodes`
  in `nemo_rl/algorithms/grpo.py`). With more than one policy node, generation must also take whole nodes:
  `policy.generation.colocated.resources.gpus_per_node` "must be explicitly set and equal to
  cluster.gpus_per_node". Final layout: trainer node (8 GPUs), vLLM node (8 GPUs, 4 × TP2), teacher node
  (1 GPU used, 7 idle). That is 24 GPUs held, against 16 for the other frameworks, which share one node
  between the teacher (1 GPU) and the policy (6 GPUs, 3 × TP2).
- **Recipes.** The reference MOPD recipe is a self-distillation smoke test (student == teacher, "OPD loss
  stays near zero"). There is no MOPD recipe for Qwen3.5/3.6, so model settings came from the
  Qwen3.5-35B-A3B GRPO recipes (Megatron EP16, Automodel EP16).
- **Runtime.** The README recommends the NGC container. `main` has no matching container, so we used the
  latest release, v0.7.0 (tag `81aa43dd`, 2026-07-29), which already contains MOPD, the Qwen3.5-35B-A3B
  recipes and a NeMo Gym pin with the `reasoning_gym` server. The image pulled anonymously from NGC.
- **Container provenance.** The image's `/opt/nemo-rl` and `NEMO_RL_COMMIT` are `19244a09` on `r0.7.0`,
  one commit before the tag, which only bumps docs and `package_info.py`. The package pins `19244a09` and
  the controller refuses to start if the container's `NEMO_RL_COMMIT` differs. Image details: 43 GB,
  `/opt/nemo_rl_venv`, prebuilt worker venvs in `/opt/ray_venvs`, Gym venvs in `/opt/gym_venvs`, Python
  3.13.13.
- **NeMo Gym.** Pinned as a submodule (`267305e2` on `main`, `d67ad661` in v0.7.0). Its `reasoning_gym`
  resources server takes rows of `responses_create_params.input` (chat messages) + `question`, `answer`,
  `metadata` (the reasoning_gym entry, with `source_dataset`) + `agent_ref`. Our packaged rows convert
  one-to-one, keeping the exact system and user messages the other frameworks train on.
- **Scoring.** The Gym server scores the final message text (`<answer>` tags, then `\boxed{}`, then the
  whole text), more leniently than our verifier. In MOPD that reward is diagnostic only.
- **Unpinned dependency.** The Gym server installs `reasoning-gym>=0.1.19` into its own venv at start-up.
  The package pins 0.1.25 with a uv constraint file (`UV_CONSTRAINT`), without editing upstream.
- **Launch.** Multi-node launch follows upstream's `ray.sub` (`docs/cluster.md`). The package controller
  runs it inside the allocation with `CONTAINER`, `MOUNTS` and `COMMAND`, and starts telemetry capture on
  each host with `srun --overlap`. `ray.sub` passes `-A $SLURM_JOB_ACCOUNT` unquoted; the value here is
  `root`, so it is harmless on this cluster.
- **Thinking.** Qwen3.6's chat template enables thinking unless `enable_thinking` is false, so Gym
  rollouts think.
- **Final weights** would come from `examples/converters/convert_megatron_to_hf.py` (documented in the
  README and `docs/about/evaluation.md`) applied to `policy/weights/iter_*`.
- Config keys were checked against v0.7.0 `examples/configs/grpo_math_1B.yaml`.

## Attempts

### Getting the container to run (jobs 985–996)

| Job | Runtime | Failure | Cause | Change |
|---|---|---|---|---|
| 985 | 84 s | Driver: `uv` "Could not acquire lock ... Read-only file system" at `/root/.cache/uv` | `ray.sub` runs containers without `--container-writable` | Set `ray.sub`'s documented `UV_CACHE_DIR_OVERRIDE` (writable uv cache); put compile caches and `HF_HOME` in the campaign |
| 986 | 146 s | `uv run` tried to rebuild the editable `nemo_rl` in `/opt/nemo-rl`: "Cannot update time stamp of directory 'nemo_rl.egg-info'" | Upstream's flow runs `uv run` and assumes a writable container; `ray.sub` has no switch for it | `uv run --no-sync` (the image's venv is built from this code) |
| 987 | 31 s | `No module named 'nemo_rl'` | The image's venv does not contain the project; upstream relies on `uv run` installing it on every launch | Put `/opt/nemo-rl` on `PYTHONPATH` for the driver and all Ray processes |
| 988 | 30 s | `cannot import name 'AutoProcessor' from 'transformers' (unknown location)` | Found in 995, below | |
| 994 | cancelled at 3 min | Ray head never started | Our mistake: we set `HOME` for the whole allocation so swanlab (imported at start-up, creates `~/.swanlab`) had a writable home. Enroot keeps container root filesystems under `$HOME/.local/share/enroot`, so the 43 GB image started extracting onto Lustre (17 GB written) | Set `HOME` for the driver only |
| 995 | | Same as 988; diagnosed with a `sys.path` probe | The image's venv files are symlinks into its own uv cache (`/opt/nemo_rl_venv/.../transformers/__init__.py -> /root/.cache/uv/archive-v0/...`). `UV_CACHE_DIR_OVERRIDE` (the 985 fix) mounts a host directory over `/root/.cache/uv`, so every package becomes a dangling symlink and imports as an empty namespace package | Drop the override mount; point `UV_CACHE_DIR` at a separate writable cache |
| 996 | | Every vLLM worker: `Read-only file system: '/opt/ray_venvs/.../vllm/v1/executor/ray_executor.py.patch_lock'` | NeMo-RL patches vLLM's installed files at run time (`nemo_rl/models/generation/vllm/patches.py`), so it needs a writable root filesystem, and `ray.sub` never passes `--container-writable` | `PYXIS_CONTAINER_WRITABLE=1` (Pyxis' environment form of the flag) fixes every `ray.sub` step without editing upstream. All earlier workarounds (`--no-sync`, `PYTHONPATH`, driver `HOME`, `UV_CACHE_DIR`) were removed; the driver runs upstream's plain `uv run` |

`ray.sub` names its containers (`--container-name=ray-head` / `ray-worker`), so Pyxis keeps the extracted
52 GB root filesystem on each node between jobs: restarts are fast, but the disk use persists.

### Job 1000: eager mode too slow

First clean start; cancelled at 31 min. Setup took 496 s (vLLM 161 s, teacher 290 s, NeMo Gym 167 s,
overlapping), then pre-training validation (two collections of about 6 min). Training rollouts took about
21 minutes each end to end: 179 rollouts finished in 30 minutes (126 of them validation). Trainer GPUs
showed 100% utilization at about 250 W, i.e. spin-waiting.

Cause: vLLM 0.20.0 ran with `cudagraph_mode=none` and no compilation, because every upstream
Qwen3.5-35B-A3B recipe sets `enforce_eager: true` and we copied it. Those recipes generate at most 4k
tokens; at 30k-token thinking, eager MoE decoding is impractically slow. Upstream documents eager mode
only in connection with a DeepScaleR convergence issue; the base config default is `enforce_eager:
False`. Torch dynamo also hit its recompile limit (8) on `_prepare_qkv_for_gated_delta_rule` (one compile
per sequence length) and fell back to eager for that function.

### Jobs 1001–1003: deadlock in the first weight sync

- **1001** (CUDA graphs on; cancelled at 24 min). Graphs captured cleanly. Pre-training validation plus
  the first training batch finished in about 15 minutes (268 rollouts, against about 50 training rollouts
  in the same time in eager mode). The driver then blocked in the first weight sync
  (`refit_policy_generation`, `grpo.py:1947`, called from `async_grpo_train` line 3553; an NCCL broadcast
  from the Megatron trainer to vLLM) with trainer, vLLM and teacher idle for 10+ minutes. Located with
  py-spy on the driver (Pyxis containers share the host PID namespace). NCCL logged nothing by default.
  The image's `LD_LIBRARY_PATH` includes AWS's EFA NCCL plugin (`/opt/amazon/ofi-nccl`), which this
  cluster does not have.
- **1002** (`NCCL_DEBUG=INFO`; cancelled at about 45 min). Same hang at the same point.
  - NCCL chose InfiniBand (`IBext_v11`, GPUDirect RDMA on the mlx5 HCAs), not EFA, and logged no
    warnings. A 16-rank, 2-node communicator (8 trainer + 8 vLLM ranks) initialized.
  - Trainer ranks (py-spy through `srun --overlap`, since Kubernetes exec to that node was failing) were
    all in `packed_broadcast_producer` (`nemo_rl/utils/packed_tensor.py:68`) → `stream.synchronize()`:
    the broadcast was issued and waiting for receivers.
  - On the vLLM node, the `VllmAsyncGenerationWorker` actors, the four `VLLM::EngineCore` processes and
    the TP workers (`RayWorkerWrapper`) were all idle. We read this as the receive half
    (`update_weights_from_collective` → `collective_rpc`) never reaching the engines, but engines that
    had already failed and returned look the same, so this was not established.
  - GPUs at idle power on all three nodes; no NCCL or Python error anywhere.
  - Upstream issues searched ("refit hang", `update_weights_from_collective`, non-colocated weight
    update): nothing matching.
- **1003** (8 × TP1 vLLM engines; cancelled at 33 min). Same deadlock after validation and the first batch
  (273 rollouts), so vLLM's multi-GPU Ray executor is not the cause. We stopped the production attempts
  here.

Ray placed the groups differently between jobs (the teacher on gpu-nodes-2 in 1000, gpu-nodes-0 in 1001),
so node roles have to be read from the logs, not the site file.

### Diagnosis (jobs 1008, 1010)

- **1008**: upstream's reference MOPD recipe (Qwen3-1.7B self-distillation) passed the first weight sync
  in about 2 minutes and started training. It also showed why no "Refitting" lines appeared in our logs:
  the package's driver ran without `PYTHONUNBUFFERED=1`.
- **1010**: the same reference recipe with only the model changed to Qwen3.6-35B-A3B (EP8) deadlocked at
  the first weight sync. With unbuffered output, every vLLM worker logs `shard_dim=0 is not a valid data
  dimension for a 3D tensor (expected 1 or 2)` in `update_weights_from_collective`. NeMo-RL catches it
  and returns `False` (`vllm_backend.py:466-470`), while the driver waits on the trainer's broadcast
  (`grpo.py:1946`) before checking vLLM's result, so it waits forever. Our log filters had dropped lines
  containing "repeated", which is how Ray tags this worker error.

`grpo.py:3553` in jobs 1001–1003 is the same start-up refit before the first update (`🔄 Refitting policy
generation with actual model weights...`) that logged the `shard_dim` error and never returned in 1010.
Background rollout collection continues during it, which is why the first batch finished. Jobs 1001–1003
logged no such error because their output was buffered.

Status: unresolved upstream. No NeMo-RL training update completed.

## Reproduction

`frameworks/nemo-rl/repro/` holds three model-free container checks (`image`, `uv-override`, `readonly`),
each about 2 minutes on one GPU. The small-model and reference-recipe checks used in 1008 and 1010 have
been removed from `repro.sh`. The deadlock is reproduced with the production configuration by
`tools/reproduce.sh nemo-rl`, whose watchdog cancels the run after `DEADLOCK_MIN` minutes in the first
sync and writes the worker errors to `DEADLOCK`. The driver now runs with `PYTHONUNBUFFERED=1`.
