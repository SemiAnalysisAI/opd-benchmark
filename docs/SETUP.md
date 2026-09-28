# Setup and execution

## Prerequisites

- Linux x86_64, Python 3.12, Slurm and a shared writable filesystem.
- Two exclusive eight-GPU B200 nodes. NeMo-RL needs a third, because its teacher reserves a whole node.
- For Miles, Slime and NeMo-RL: Pyxis/Enroot with `--container-image`, root remapping and writable containers. Prime-RL and verl run from virtual environments on the hosts.
- Each node requests 80 CPU cores and exposes 64 to Ray.

This package does not install Slurm, change cluster services or build a container image.

Site file rules, enforced by `tools/prepare.py`:

- For Miles and Slime, the trainer IPv4 address must sort before the generation address. Both also check the Ray GPU placement before training.
- `base_model`, `megatron_model` and `teachers` must be under `assets`.
- Paths must be absolute and contain no spaces or shell metacharacters. The campaign and asset paths must be the same on every node.

Replace the documentation-only addresses in `config/site.example.json` with routable ones. Do not expose Ray or model endpoints publicly.

## One command: `tools/reproduce.sh`

`tools/reproduce.sh` runs every step in this guide for the single-teacher `caesar_cipher` benchmark in the [report](CAESAR-OPD-2026-09-27.md), then benchmarks each saved checkpoint and writes the telemetry report.

```bash
PARTITION=YOUR_PARTITION tools/reproduce.sh all                 # references, every framework, then the report
PARTITION=YOUR_PARTITION tools/reproduce.sh verl slime          # chosen frameworks
PARTITION=YOUR_PARTITION tools/reproduce.sh references report   # base and teacher benchmarks; telemetry summary
```

Run it on a Slurm login node with Pyxis/Enroot, after filling in `config/site.local.json`. For NeMo-RL, also copy `config/site-nemo-rl.example.json` to `config/site-nemo-rl.local.json` and fill it in.

For each framework the script:

1. downloads the models and fuses the teacher;
2. prepares `$RUNS/<framework>-$NAME`;
3. imports the pinned container, converts the Megatron checkpoint (Miles and Slime) and builds the runtime: pydeps with the runtime's own Python, Slime's wheels, the uv venvs and prebuilt FlashInfer;
4. submits the run and blocks until the allocation ends;
5. exports Hugging Face weights of each saved checkpoint and benchmarks them.

A step is skipped when its output exists, so rerunning resumes. Settings (`RUNS`, `RUNTIME`, `NAME`, `GPU_NODE`, `BENCH_CAMPAIGN`, `DEADLOCK_MIN` and the site files) are listed at the top of the script.

For NeMo-RL the script also runs a watchdog. If the first trainer-to-vLLM weight sync has not completed `DEADLOCK_MIN` minutes (default 20) after it starts, the watchdog writes the vLLM workers' weight-update errors to the run's `DEADLOCK` file and cancels the job. Otherwise the job would hold 24 GPUs until its time limit. See [the NeMo-RL reproduction](../frameworks/nemo-rl/repro/README.md).

What has been tested:

- A resume over this repository's finished runs (`NAME=caesar-01`): every step was skipped and the report was regenerated identically.
- `tools/reproduce.sh miles` into a new campaign (`NAME=caesar-02`, job 1021): it prepared the campaign, built its pydeps, ran 20 updates, exported the Hugging Face weights and benchmarked them unattended. `BENCH_CAMPAIGN` pointed it at an existing verl venv.

The downloads, container imports, `torch_dist` conversion and venv builds use the commands in this guide, but the script has not yet run them from an empty `RUNS`.

## Models

Revisions are pinned in `shared/recipe.py`. Weights are not stored here. Keep any Hugging Face token outside the repository and the site file.

| Role | Hugging Face repository | Revision |
|---|---|---|
| Base | `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| `caesar_cipher` teacher | `semianalysisai/Qwen3.6-35B-A3B-caesar-cipher-GRPO-20260924` | `dfedf677a5bf08a451733ec48c845c2661a6ab14` (tag `step-125`) |
| `simple_geometry` teacher | `semianalysisai/Qwen3.6-35B-A3B-simple-geometry-GRPO-20260925` | `679d4b66018c03249a3bfd0ecca3195ed9cd0e0d` (tag `step-50`) |

```bash
hf download Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --local-dir /shared/opd-assets/models/Qwen3.6-35B-A3B
hf download semianalysisai/Qwen3.6-35B-A3B-caesar-cipher-GRPO-20260924 --revision dfedf677a5bf08a451733ec48c845c2661a6ab14 --local-dir /shared/opd-assets/hub/caesar_cipher
hf download semianalysisai/Qwen3.6-35B-A3B-simple-geometry-GRPO-20260925 --revision 679d4b66018c03249a3bfd0ecca3195ed9cd0e0d --local-dir /shared/opd-assets/hub/simple_geometry
```

The Hub teachers use Prime-RL's per-expert layout. Every framework serves copies fused into the base model's layout, with the base model's MTP tensors. The site's `teachers` directory must hold one fused copy per trained task (`caesar_cipher/`, `simple_geometry/`). `tools/fuse_teacher.py` needs torch and safetensors:

```bash
python3 tools/fuse_teacher.py /shared/opd-assets/hub/caesar_cipher /shared/opd-assets/teachers/caesar_cipher /shared/opd-assets/models/Qwen3.6-35B-A3B
python3 tools/fuse_teacher.py /shared/opd-assets/hub/simple_geometry /shared/opd-assets/teachers/simple_geometry /shared/opd-assets/models/Qwen3.6-35B-A3B
```

The teachers are frozen. Use the included data; do not regenerate it.

## Prepare a campaign

```bash
cp config/site.example.json config/site.local.json   # Edit before continuing.
python3 tools/prepare.py slime --site config/site.local.json --output /shared/opd-runs/slime-01
```

Add `--domains caesar_cipher` for single-teacher OPD on one task. This writes the campaign's `experiment.json`.

Preparation:

- checks out the pinned upstream revision and applies the checked patch, if any;
- verifies every dataset hash;
- copies `shared/` and the framework's files, and links the models;
- writes `site.json`, which campaign processes read at runtime, and `package-provenance.json`.

It does no model conversion, environment installation or submission. If it fails, use a fresh output path.

Every framework except NeMo-RL scores with `reasoning-gym==0.1.25` from the campaign's `pydeps/`. Install it with a Python 3.12 Linux x86_64 interpreter that matches the runtime (the container's Python for Miles and Slime, the venv's for Prime-RL and verl):

```bash
python3 -m pip install --target /shared/opd-runs/slime-01/pydeps reasoning-gym==0.1.25
```

`$CAMPAIGN_PYDEPS` appends this directory to `sys.path`, so runtime packages take precedence.

## Miles and Slime runtime

`tools/reproduce.sh` imports the digest-pinned image `radixark/miles@sha256:59a11219eae0defc6594ec678fafe4e897c16904263223f79968cd3e0209a502` to the site's `container_image`. The September 27 runs used it; its Megatron revision matches the recorded one below. Set `megatron_source` to Megatron's path inside the container.

The previous experiment used an image named `miles-dev-202609050049.sqsh`. Its bytes are not distributed, and the name is not a digest. It contained:

- PyTorch 2.13.0+cu130, SGLang 0.5.19.dev56+ga359142 (source `a3591427c20431d33570f0c766d0cee6711f5ee8`), Ray 2.58.0, Transformers 5.12.1
- Transformer Engine 2.17.0, Flash Linear Attention 0.5.2, SGLang Router 0.3.2
- Megatron `8c1e05747eb612b382df2632783df5c83a853646`

Installing these versions does not necessarily recreate its CUDA binaries, because the image used locally compiled wheels. Slime's observed inventory is in `frameworks/slime/upstream/recorded-runtime.json`.

Both frameworks need a Megatron `torch_dist` base checkpoint at `megatron_model`, linked into the campaign as `models/Qwen3.6-35B-A3B_torch_dist`. Use the pinned Miles converter (`tools/convert_hf_to_torch_dist.py`), as `tools/reproduce.sh` does. The recorded runs reused an existing conversion.

Slime installs NumPy 1.26.4 and SciPy 1.15.3 in each container from checked wheels. Download them into the campaign first:

```bash
python3 -m pip download --only-binary=:all: --no-deps --platform manylinux2014_x86_64 --python-version 312 --implementation cp --abi cp312 --dest /shared/opd-runs/slime-01/wheels numpy==1.26.4 scipy==1.15.3
cd /shared/opd-runs/slime-01 && sha256sum -c wheels.sha256
```

Slime uses native NCCL groups. This is safe only because trainer offload and release are disabled.

## Prime-RL runtime

Prime-RL needs a working environment at the campaign's `source/.venv`. Preparation initializes the pinned submodules over HTTPS. Build the environment from the pinned lock, then install the task environment:

```bash
cd /shared/opd-runs/prime-01/source && uv sync --frozen --python 3.12 --all-extras
uv pip install --python /shared/opd-runs/prime-01/source/.venv/bin/python --no-deps -e /shared/opd-runs/prime-01/environments/rg_tasks
```

Use the `uv` path from your site file. Runs use `--no-sync`, so submission never replaces the environment. The previous experiment used PyTorch 2.13.0+cu130 and vLLM 0.28.0; its inventory is `frameworks/prime-rl/upstream/recorded-packages.txt`.

On its first run, Prime-RL's trainer converts the base model to its native layout in `base_model/prime`, so the base model directory must be writable.

## verl runtime

verl runs unmodified from the campaign's `source/.venv`. It is built from the pinned `uv.lock` with the extras of upstream `examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_fsdp.sh`. It trains with FSDP and needs no Megatron conversion.

```bash
cd /shared/opd-runs/verl-01/source && uv sync --frozen --python 3.12 --all-packages --extra vllm --extra fsdp
```

## NeMo-RL runtime

NeMo-RL runs unmodified from `nvcr.io/nvidia/nemo-rl:v0.7.0`, imported once with Enroot (43 GB) to the site's `nemo_rl_container`:

```bash
enroot import -o /shared/opd-runtime/nemo-rl-v0.7.0.sqsh 'docker://nvcr.io#nvidia/nemo-rl:v0.7.0'
```

The image is built from `19244a09` on `r0.7.0`, one commit before the `v0.7.0` tag. The controller refuses to start if the container's `NEMO_RL_COMMIT` differs. NeMo-RL needs no pydeps or conversion: NeMo Gym scores the rollouts, and the package pins its `reasoning-gym` to 0.1.25 with a uv constraint.

The controller launches upstream's `ray.sub` inside the allocation with `PYXIS_CONTAINER_WRITABLE=1`, because NeMo-RL writes into its image at runtime. Do not set `ray.sub`'s `UV_CACHE_DIR_OVERRIDE`; it hides the image's venv.

No NeMo-RL run has completed a training update. Every run deadlocked at the first weight sync.

## Prebuilt FlashInfer kernels (Prime-RL and verl)

Both locks install only `flashinfer-python`, which compiles or downloads kernels on first use. On B200 the MoE kernel alone leaves the GPUs idle for about 15 minutes. Install FlashInfer's prebuilt packages at exactly the venv's `flashinfer-python` version (0.6.16.post3 for Prime-RL, 0.6.18 for verl), then check `flashinfer show-config`:

```bash
V=0.6.18; PY=/shared/opd-runs/verl-01/source/.venv/bin/python
uv pip install --python $PY --no-deps --index-url https://flashinfer.ai/whl/ flashinfer-cubin==$V
uv pip install --python $PY --no-deps --index-url https://flashinfer.ai/whl/cu130/ flashinfer-jit-cache==$V+cu130
```

Every framework keeps its other compiled kernels (Triton, Inductor, vLLM, SGLang) in the campaign's `runtime-cache/`, so only a campaign's first run compiles them.

## Submit and monitor

```bash
python3 tools/submit.py /shared/opd-runs/slime-01 --partition YOUR_PARTITION            # Prints the request.
python3 tools/submit.py /shared/opd-runs/slime-01 --partition YOUR_PARTITION --submit   # Allocates 16 GPUs.
```

Run it on a Slurm login host or pod; there is no built-in Kubernetes support. It runs `python3 <campaign>/node.py` under `salloc`. `node.py` is the controller: it starts `node.py learner` and `node.py generation` on the two nodes.

- The command stays attached and holds a per-campaign lock. It refuses a new run while the previous one has no terminal record.
- The default time limit is 4 hours (`--time 04:00:00`).
- The controller stops its processes when training ends or fails. It never retries or resubmits.

There is no automated preflight. `tools/submit.py` only checks that required files exist (models, teachers, pydeps, venv or container, and Slime's wheels). Validate the runtime, conversions and teacher endpoints yourself, for example with a short run, before a full experiment.

The job number is in `active-run.json`. Each run writes `results/<framework>-<job>/`, including `start.json`, `end.json`, `exit-code.txt`, `launch-source.tar.gz` and `<role>-capture.jsonl.gz` (GPU, host and Prometheus samples).

Driver logs:

| Framework | Log |
|---|---|
| Miles, Slime, verl | `results/<run>/learner-train.out` |
| Prime-RL | Component logs in the directory named by `resolved-paths.json` |
| NeMo-RL | `results/<run>/*-logs/ray-driver.log` |

Results can contain full prompts and responses; review them before sharing. Cancel a run with `scancel YOUR_JOB_ID`.

## Known limits

Only 16 B200 GPUs (24 for NeMo-RL) have been tested; no other GPU type or node count is supported. `tools/reproduce.sh` has not yet run on a fresh machine. Validate on GPUs before treating a new image or hardware configuration as equivalent.
