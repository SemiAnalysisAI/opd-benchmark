# Setup and execution

## Prerequisites

The layout uses Linux x86_64, Python 3.12, Slurm, a shared writable filesystem, and two exclusive eight-GPU B200 nodes.
Miles and Slime also require Pyxis/Enroot support for `--container-image`, root remapping, and writable containers.
Prime-RL runs from a virtual environment on the hosts. Every role requests 80 CPU cores per node; the container roles expose 64 to Ray.
NeMo-RL needs a third node (its teacher reserves a whole node) and runs from NVIDIA's `nvcr.io/nvidia/nemo-rl:v0.7.0` container.
This package does not install Slurm, modify cluster services, or create a container image.

For Miles and Slime, the trainer IPv4 address must sort before the generation address.
`tools/prepare.py` rejects other orderings. Miles and Slime check the Ray GPU placement before training.
Replace the documentation-only addresses in `config/site.example.json` with routable addresses. Do not expose Ray or model endpoints publicly.
Model paths must be under `assets`. The campaign and asset paths must be identical on both nodes and contain no spaces or shell metacharacters.

## One command: `tools/reproduce.sh`

`tools/reproduce.sh` runs every step below for the single-teacher caesar_cipher benchmark of the
[report](CAESAR-OPD-2026-09-27.md), with the production setup (Qwen3.6-35B-A3B, the caesar_cipher teacher,
the shared recipe), then benchmarks each final model and writes the telemetry report:

```bash
PARTITION=YOUR_PARTITION tools/reproduce.sh all                 # references, then every framework, then the report
PARTITION=YOUR_PARTITION tools/reproduce.sh verl slime          # chosen frameworks
PARTITION=YOUR_PARTITION tools/reproduce.sh references report   # base and teacher benchmarks; telemetry summary
```

Run it on a Slurm login node with Pyxis/Enroot, after filling in `config/site.local.json` and, for NeMo-RL,
`config/site-nemo-rl.local.json` (copy `config/site-nemo-rl.example.json`). For each framework it downloads and
fuses the models, imports the pinned container, converts the Megatron checkpoint (Miles and Slime), builds the
runtime (pydeps with the runtime's own Python, Slime's wheels, the uv venvs, prebuilt FlashInfer), prepares
`$RUNS/<framework>-caesar`, submits the run and blocks until its allocation ends, exports the final Hugging Face
weights and benchmarks them. A step is skipped when its output exists, so re-running resumes. The settings
(`RUNS`, `RUNTIME`, `NAME`, `GPU_NODE`, `DEADLOCK_MIN`, the site paths) are listed at the top of the script.

For NeMo-RL it also runs a watchdog: if the first trainer-to-vLLM weight sync has not completed `DEADLOCK_MIN`
minutes (default 20) after it starts, the watchdog writes the vLLM workers' weight-update errors to the run's
`DEADLOCK` file and cancels the job, which would otherwise hold 24 GPUs until its time limit (see
[the NeMo-RL reproduction](../frameworks/nemo-rl/repro/README.md)).

What has been exercised: a resume over this repository's finished runs (`NAME=caesar-01`, every step
skipped, the report regenerated identically), and `tools/reproduce.sh miles` into a new campaign
(`NAME=caesar-02`, job 1021), which prepared the campaign, built its pydeps, ran 20 updates, exported the
Hugging Face weights and benchmarked them unattended. `BENCH_CAMPAIGN` pointed it at an existing verl venv.
The downloads, container imports, torch_dist conversion and venv builds are the commands this guide
documents, but the script has not yet run them from an empty `RUNS`.

## Models

The revisions are in `shared/recipe.py`. Weights are not stored here. Keep any Hugging Face token outside the repository and site file.

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

The Hub teachers use Prime-RL's per-expert layout. All three frameworks serve copies fused into the base's layout, with the base's MTP tensors.
The site's `teachers` directory must hold `caesar_cipher/` and `simple_geometry/` fused copies. The tool needs torch and safetensors.

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

Add `--domains caesar_cipher` for single-teacher OPD on one task; it writes the campaign's `experiment.json`.

Preparation checks out the recorded upstream revision, applies the checked patch, and verifies every dataset hash.
It copies `shared/` and the framework's files verbatim, links the models, and writes `site.json` and `package-provenance.json`.
Campaign processes read `site.json` at runtime. Preparation does no model conversion, environment installation, or submission.
If it fails, use a fresh output path.

The scorer needs `reasoning-gym==0.1.25` in the campaign's `pydeps/`. Install it with a Python 3.12 Linux x86_64 interpreter that matches the runtime:

```bash
python3 -m pip install --target /shared/opd-runs/slime-01/pydeps reasoning-gym==0.1.25
```

`$CAMPAIGN_PYDEPS` puts this directory at the end of `sys.path`, so runtime packages take precedence.

## Miles and Slime runtime

The previous experiment reused a container identified as `miles-dev-202609050049.sqsh`. Its bytes are not distributed, and its name is not a digest.
It contained PyTorch 2.13.0+cu130, SGLang 0.5.19.dev56+ga359142, Ray 2.58.0, Transformers 5.12.1,
Transformer Engine 2.17.0, Flash Linear Attention 0.5.2, and SGLang Router 0.3.2.
SGLang's source revision was `a3591427c20431d33570f0c766d0cee6711f5ee8`; Megatron's was `8c1e05747eb612b382df2632783df5c83a853646`.
Set `megatron_source` to Megatron's path inside the container.
Installing these versions does not necessarily recreate the original CUDA binaries, because the image used local compiled wheels.
Slime's observed inventory is in `frameworks/slime/upstream/recorded-runtime.json`.

Both require a converted Megatron `torch_dist` base checkpoint at `megatron_model`, linked as `models/Qwen3.6-35B-A3B_torch_dist`.
The recorded runs reused an existing conversion; no fresh conversion is claimed. Use the pinned upstream converter.

Slime installs NumPy 1.26.4 and SciPy 1.15.3 in each container from checked wheels. Download them into the campaign first:

```bash
python3 -m pip download --only-binary=:all: --no-deps --platform manylinux2014_x86_64 --python-version 312 --implementation cp --abi cp312 --dest /shared/opd-runs/slime-01/wheels numpy==1.26.4 scipy==1.15.3
cd /shared/opd-runs/slime-01 && sha256sum -c wheels.sha256
```

Slime uses native NCCL groups, which is safe only because trainer offload and release are disabled.

## Prime-RL runtime

Prime-RL requires a working environment at the campaign's `source/.venv`. The previous experiment used PyTorch 2.13.0+cu130 and vLLM 0.28.0.
Its inventory is `frameworks/prime-rl/upstream/recorded-packages.txt`. Preparation initializes the pinned submodules over HTTPS.
Install the pinned source's dependencies, then the task environment:

```bash
/usr/local/bin/uv pip install --python /shared/opd-runs/prime-01/source/.venv/bin/python --no-deps -e /shared/opd-runs/prime-01/environments/rg_tasks
```

Use the `uv` path from your site file. Runs use `--no-sync`, so submission never replaces the environment.
On its first run, Prime-RL's trainer converts the base model to its native layout in `base_model/prime`, so the base model directory must be writable.

## verl runtime

verl runs unmodified from the campaign's `source/.venv`, built from the pinned `uv.lock` with the extras of upstream
`examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_fsdp.sh`. It trains with FSDP and needs no Megatron conversion.

```bash
cd /shared/opd-runs/verl-01/source && uv sync --frozen --all-packages --extra vllm --extra fsdp
```

Follow `results/*/learner-train.out` for the driver log.

## NeMo-RL runtime

NeMo-RL runs unmodified from `nvcr.io/nvidia/nemo-rl:v0.7.0`, imported once with Enroot (43 GB) to the site's
`nemo_rl_container`. The image is built from `19244a09` on `r0.7.0`, one commit before the `v0.7.0` tag, and the
controller refuses to start if the container's `NEMO_RL_COMMIT` differs. It needs no pydeps or conversion:
NeMo Gym scores the rollouts, and the package pins its `reasoning-gym` to 0.1.25 with a uv constraint.

```bash
enroot import -o /shared/opd-runtime/nemo-rl-v0.7.0.sqsh 'docker://nvcr.io#nvidia/nemo-rl:v0.7.0'
```

The controller launches upstream's `ray.sub` inside the allocation with `PYXIS_CONTAINER_WRITABLE=1`, because
NeMo-RL writes into its image at run time; do not set `ray.sub`'s `UV_CACHE_DIR_OVERRIDE`, which hides the
image's venv. Follow `results/*/ray-driver.log` (under the run's `*-logs/` directory) for the driver log. No run
of ours has completed a NeMo-RL training update: every run deadlocked at the first weight sync.

## Prebuilt FlashInfer kernels (Prime-RL and verl)

Both locks install only `flashinfer-python`, which compiles or downloads kernels on first use; on B200 the MoE kernel
alone takes about 15 minutes of idle GPUs. Install FlashInfer's prebuilt packages at exactly the venv's
`flashinfer-python` version (0.6.16.post3 for Prime-RL, 0.6.18 for verl), then check `flashinfer show-config`:

```bash
V=0.6.18; PY=/shared/opd-runs/verl-01/source/.venv/bin/python
uv pip install --python $PY --no-deps --index-url https://flashinfer.ai/whl/ flashinfer-cubin==$V
uv pip install --python $PY --no-deps --index-url https://flashinfer.ai/whl/cu130/ flashinfer-jit-cache==$V+cu130
```

Every framework keeps its remaining compiled kernels (Triton, Inductor, vLLM, SGLang) in the campaign's
`runtime-cache/`, so only a campaign's first run compiles them.

## No automated preflight

The campaigns run no contract tests before training. `tools/submit.py` only checks that the required files exist, including `pydeps/reasoning_gym`.
Validate the runtime, conversions, and teacher endpoints yourself, for example with a short run, before a full experiment.

## Submit and monitor

```bash
python3 tools/submit.py /shared/opd-runs/slime-01 --partition YOUR_PARTITION            # Prints the request.
python3 tools/submit.py /shared/opd-runs/slime-01 --partition YOUR_PARTITION --submit   # Allocates 16 GPUs.
```

Run it on a Slurm login host or pod; no Kubernetes context is built in. It runs `python3 <campaign>/node.py` under `salloc`.
`node.py` is the controller; it starts `node.py learner` and `node.py generation` on the two nodes.
The command stays attached, holds a per-campaign lock, and refuses a new run while the previous one lacks a terminal record.
The default time limit is 1h30m (`--time`). The controller stops its processes when training ends or fails. It never retries or resubmits.

The job number is in `active-run.json`. Each run writes `results/<framework>-<job>/`, including `start.json`, `end.json`, `exit-code.txt`,
`launch-source.tar.gz`, and `<role>-capture.jsonl.gz` (GPU, host, and Prometheus samples).
For Miles and Slime, follow `learner-train.out`. For Prime-RL, follow the component logs in the directory named by `resolved-paths.json`.
Results can contain full prompts and responses. Review them before sharing. Use `scancel YOUR_JOB_ID` to cancel.

## Known limits

This package does not establish support for any GPU or node count other than 16 B200 GPUs (24 for NeMo-RL).
`tools/reproduce.sh` automates the runtime and conversion steps, but it has not yet been run on a fresh machine.
Validate on GPUs before calling a new image or hardware configuration equivalent.
