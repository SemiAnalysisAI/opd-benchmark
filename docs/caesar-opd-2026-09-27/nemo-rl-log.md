# NeMo-RL onboarding log (working notes for the report)

Upstream: `NVIDIA-NeMo/RL` at `4aaa48fabd178a4bf481e52c49e9125995e0f2a8` (main, 2026-09-24).
Chosen mode: MOPD on async GRPO (`adv_estimator: opd`), rollouts through NeMo Gym, per
`docs/about/algorithms/mopd.md` and `examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack.yaml`.

## Reading the docs (before running anything)

- Two distillation paths exist and the docs keep them apart clearly:
  `run_distillation.py` (teacher-logit KD, "currently supports the DTensor and vLLM generation
  backend. Megatron generation/training paths are not supported yet") and MOPD (async GRPO with a
  sampled-token teacher-minus-student advantage, the same objective as the other four frameworks).
  The on-policy-distillation guide's Megatron note conflicts with a checked-in
  `distillation_math_megatron.yaml`; not investigated because MOPD is the path we use.
- MOPD teachers are non-colocated and reserved in whole nodes: `opd_teacher_nodes += num_nodes`,
  and the policy gets the remaining nodes (`nemo_rl/algorithms/grpo.py`). A 1-GPU teacher still takes
  a node, so the smallest MOPD layout is 3 nodes (trainer, vLLM, teacher), where the other four
  frameworks share one node between the teacher (1 GPU) and the policy (6 GPUs).
- MOPD requires NeMo Gym rollouts. NeMo Gym (pinned `267305e2` as a submodule) ships a
  `reasoning_gym` resources server, so the task maps directly.
- The reference MOPD recipe is a self-distillation smoke test (student == teacher, "OPD loss stays
  near zero"); there is no MOPD recipe for Qwen3.5/3.6. There are Qwen3.5-35B-A3B GRPO recipes
  (Megatron EP16, Automodel EP16) to take the model-specific settings from.
- With more than one policy node, generation must take whole nodes too:
  `policy.generation.colocated.resources.gpus_per_node` "must be explicitly set and equal to
  cluster.gpus_per_node" (`grpo.py`). So the policy gets 8 vLLM GPUs (4 × TP2), not the 6 (3 × TP2)
  the other frameworks use. Final layout: trainer node (8 GPUs), vLLM node (8), teacher node (teacher on
  1 GPU, as elsewhere; 7 idle). 24 GPUs held versus 16.
- Runtime: the README recommends the NGC container. `main` has no matching container, so we pinned the
  latest release, v0.7.0 (`81aa43dd`, 2026-07-29), which already contains MOPD, the Qwen3.5-35B-A3B
  recipes and a NeMo Gym pin (`d67ad661`) with the `reasoning_gym` server, and its container
  `nvcr.io/nvidia/nemo-rl:v0.7.0` (anonymous pull from NGC worked).
- NeMo Gym data: rows are `responses_create_params.input` (chat messages) + `question`, `answer`,
  `metadata` (the reasoning_gym entry, with `source_dataset`) + `agent_ref`. Our packaged rows convert
  one-to-one, keeping the exact system + user messages the other frameworks train on.
- The Gym `reasoning_gym` server scores the final message text (`<answer>` tags, then `\boxed{}`, then
  the whole text) — more lenient than our verifier. In MOPD that reward is diagnostic only.
- The Gym `reasoning_gym` server installs `reasoning-gym>=0.1.19` into its own venv at start-up (unpinned,
  so it would take the newest release). The package pins 0.1.25 with a uv constraint file
  (`UV_CONSTRAINT`), without editing upstream.
- Multi-node launch follows upstream's `ray.sub` (docs/cluster.md): the package controller runs it inside
  the allocation with `CONTAINER`, `MOUNTS`, `COMMAND`, and starts its own telemetry capture on each
  host with `srun --overlap`. `ray.sub` passes `-A $SLURM_JOB_ACCOUNT` unquoted; on this cluster it is
  `root`, so it is safe here.
- Qwen3.6's chat template enables thinking unless `enable_thinking` is false, so Gym rollouts think.
- Final weights: `examples/converters/convert_megatron_to_hf.py` (documented in the README and
  docs/about/evaluation.md) from `policy/weights/iter_*`.
- Config keys were checked against v0.7.0 `examples/configs/grpo_math_1B.yaml` before any GPU time.
- The `nvcr.io/nvidia/nemo-rl:v0.7.0` container is not built from the `v0.7.0` tag: its `/opt/nemo-rl`
  and `NEMO_RL_COMMIT` are `19244a09` on `r0.7.0`, one commit before the tag (`81aa43dd`), which only
  bumps docs and `package_info.py`. The package pins `19244a09` and the controller refuses to start if
  the container's `NEMO_RL_COMMIT` differs. (43 GB image; `/opt/nemo_rl_venv`, prebuilt worker venvs in
  `/opt/ray_venvs`, Gym venvs in `/opt/gym_venvs`, Python 3.13.13.)
- Our own mistake: a wait loop used `pgrep -f "enroot import"`, which matched its own command line, so
  the import finished at 21:12 but we noticed only at 22:40.

## Attempts

- **Job 985** (84 s): Ray head up, driver failed: `uv` "Could not acquire lock ... Read-only file system"
  at `/root/.cache/uv`. `ray.sub` runs the container without `--container-writable`; its documented
  switch `UV_CACHE_DIR_OVERRIDE` mounts a writable uv cache. The package now sets it, and puts the
  compile caches and `HF_HOME` in the campaign for every Ray process.
- **Job 986** (146 s): `uv run` then tried to rebuild the editable `nemo_rl` package in `/opt/nemo-rl`
  ("Cannot update time stamp of directory 'nemo_rl.egg-info'"), also read-only. Upstream's cluster flow
  runs `uv run` from a writable checkout and evidently assumes containers are writable by default;
  `ray.sub` has no switch for it. Fix: `uv run --no-sync`, since the image's venv is already built from
  exactly this code.
- **Job 987** (31 s): `No module named 'nemo_rl'`. The image's venv does not contain the project itself;
  upstream's container flow relies on `uv run` installing it editable on every launch. With `--no-sync`
  the package puts `/opt/nemo-rl` on `PYTHONPATH` for the driver and every Ray process instead.
- **Job 988** (30 s): `cannot import name 'AutoProcessor' from 'transformers' (unknown location)`, which
  a standalone container probe with the same variables could not reproduce (investigating).
- **Job 994** (our mistake, cancelled at 3 min): to give swanlab (imported at start-up, creates
  `~/.swanlab`) a writable home we set `HOME` for the whole allocation. Enroot keeps each container's
  root filesystem under `$HOME/.local/share/enroot`, so the 43 GB image started extracting onto Lustre
  (17 GB written) and the Ray head never started. `HOME` is now set for the driver only.
- **Root cause of job 988** (found in job 995 with a `sys.path` probe): the NGC image's venv files are
  symlinks into the image's own uv cache (`/opt/nemo_rl_venv/.../transformers/__init__.py ->
  /root/.cache/uv/archive-v0/...`, uv's symlink link-mode at build time). `ray.sub`'s own
  `UV_CACHE_DIR_OVERRIDE` switch (which we had used to fix job 985) mounts a host directory over
  `/root/.cache/uv`, hiding that cache, so every package in the venv becomes a dangling symlink and
  imports as an empty namespace package. Fix: no override mount; `UV_CACHE_DIR` points uv at a separate
  writable cache in the campaign.
- `ray.sub` names its containers (`--container-name=ray-head` / `ray-worker`), so Pyxis keeps the
  extracted 52 GB root filesystem on each node between jobs: fast restarts, but disk that persists.
- **Job 996**: the driver and vLLM workers started, then every vLLM worker failed:
  `Read-only file system: '/opt/ray_venvs/.../vllm/v1/executor/ray_executor.py.patch_lock'`. NeMo-RL
  patches vLLM's installed source files at runtime (`nemo_rl/models/generation/vllm/patches.py`), so
  its container flow needs a writable root filesystem, and `ray.sub` never passes
  `--container-writable` (upstream clusters presumably default to writable containers). Pyxis'
  environment form `PYXIS_CONTAINER_WRITABLE=1` fixes every `ray.sub` step without editing upstream,
  and makes the earlier workarounds (`--no-sync`, `PYTHONPATH`, driver `HOME`, `UV_CACHE_DIR`)
  unnecessary; they were removed and the driver is back to upstream's plain `uv run`.
- **Job 1000** (cancelled at 31 min): the first clean start. Setup 496 s (vLLM 161 s, teacher 290 s,
  NeMo Gym 167 s, overlapping), then the pre-training validation (two collections of about 6 min).
  Training rollouts then took about 21 minutes each end to end; 179 rollouts finished in 30 minutes
  (126 of them validation), and the trainer GPUs sat at 100% "utilization" but ~250 W (spin-waiting).
  Cause: vLLM 0.20.0 ran with `cudagraph_mode=none` and no compilation, because every upstream
  Qwen3.5-35B-A3B recipe sets `enforce_eager: true` (we copied it). Those recipes generate at most 4k
  tokens; at 30k-token thinking, eager MoE decoding is impractically slow. Upstream documents eager
  mode only for a DeepScaleR convergence issue; the base config default is `enforce_eager: False`.
  Torch dynamo also hit its recompile limit (8) on `_prepare_qkv_for_gated_delta_rule` (one compile
  per sequence length) and fell back to eager for it.
- **Job 1001** (CUDA graphs on; cancelled at 24 min): graphs captured cleanly and the pre-training
  validation plus the first training batch finished in about 15 minutes (268 rollouts, versus about 50
  training rollouts in the same time in eager mode). The driver then blocked in the first weight sync
  (`refit_policy_generation`, `grpo.py:1947` from `async_grpo_train` line 3553, an NCCL broadcast from
  the Megatron trainer to vLLM), with trainer, vLLM and teacher all idle for 10+ minutes. Found with
  py-spy on the driver (Pyxis containers share the host PID namespace). NCCL logged nothing by default;
  the image's `LD_LIBRARY_PATH` includes AWS's EFA NCCL plugin (`/opt/amazon/ofi-nccl`), which this
  cluster does not have. Relaunched with `NCCL_DEBUG=INFO`.
- Ray placed the groups differently between jobs 1000 and 1001 (teacher on gpu-nodes-2 then gpu-nodes-0),
  so node roles must be read from the logs, not the site file.
- **Job 1002** (NCCL logging on; cancelled at ~45 min): same hang at the same point. Evidence:
  - NCCL chose InfiniBand (`IBext_v11`, GPUDirect RDMA on the mlx5 HCAs), not AWS EFA, and logged no
    warnings; a 16-rank, 2-node communicator (8 trainer + 8 vLLM ranks) initialized.
  - Trainer ranks (on gpu-nodes-3, dumped with py-spy through `srun --overlap` because Kubernetes
    exec to that node was failing): all in `packed_broadcast_producer` (`nemo_rl/utils/packed_tensor.py:68`)
    → `stream.synchronize()`, i.e. the NCCL broadcast was issued and waits for receivers.
  - vLLM side (gpu-nodes-0): `VllmAsyncGenerationWorker` actors idle, the four `VLLM::EngineCore`
    processes idle on an empty input queue, the TP workers (`RayWorkerWrapper`) idle in Ray
    compiled-graph channel reads. The receive half of the weight update
    (`update_weights_from_collective` → `collective_rpc`) never reached the engines.
  - GPUs at idle power on all three nodes; no NCCL or Python error anywhere.
  - Upstream issues searched ("refit hang", `update_weights_from_collective`, non-colocated weight
    update): nothing matching.
- **Job 1003** (8 × TP1 vLLM engines; cancelled at 33 min): same deadlock in the first weight sync
  (`refit_policy_generation`, `grpo.py:1947`) after the validation and first batch (273 rollouts). So
  vLLM's multi-GPU Ray executor is not the cause. Stopped here; no NeMo-RL training update completed.
- **Repro kit** (`frameworks/nemo-rl/repro/`): the `image`, `uv-override` and `readonly` checks each
  reproduce in about 2 minutes on one GPU. Upstream's own reference MOPD recipe (Qwen3-1.7B
  self-distillation, job 1008) passed the first weight sync in about 2 minutes and started training, so
  the deadlock is specific to something in our configuration. That run also showed why the "Refitting"
  lines never appeared in our logs: the package's driver ran without `PYTHONUNBUFFERED=1`.
- **Job 1010** (`repro.sh hang-moe`: upstream's reference MOPD recipe with only the model changed to
  Qwen3.6-35B-A3B, EP8): deadlocked at the first weight sync. With unbuffered output the cause is
  visible: every vLLM worker logs `shard_dim=0 is not a valid data dimension for a 3D tensor (expected 1
  or 2)` in `update_weights_from_collective`; NeMo-RL catches it and returns `False`
  (`vllm_backend.py:466-470`), while the driver waits on the trainer's broadcast
  (`grpo.py:1946`) before checking vLLM's result, so it waits forever. Our monitors had filtered out
  lines containing "repeated", which is how Ray tags this worker error. Jobs 1001-1003 logged no such
  error in their (buffered) output; see the next entry.
- **Same call site.** `grpo.py:3553` in the production stacks (jobs 1001-1003) is the start-up refit
  before the first update (`🔄 Refitting policy generation with actual model weights...`), the call that
  logged the `shard_dim` error and never returned in job 1010. Background rollout collection continues
  during it, which is why the first batch finished. Idle vLLM engines in py-spy are also what engines that
  had already failed and returned look like, so "never reached the engines" was not established either.
- **Repro kit reduced to production.** Jobs 1008 and 1010 used small-model and reference-recipe checks
  that have since been removed from `repro.sh`; the kit now holds the three model-free container checks,
  and the deadlock is reproduced with the production configuration by `tools/reproduce.sh nemo-rl`,
  whose watchdog cancels the run after `DEADLOCK_MIN` minutes in that sync and writes the worker errors
  to `DEADLOCK`. The driver now runs with `PYTHONUNBUFFERED=1`.
