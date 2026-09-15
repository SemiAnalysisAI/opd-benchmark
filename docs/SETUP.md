# Setup and execution

## Supported layout and prerequisites

The recorded layout uses Linux x86_64, Python 3.12, Slurm, a shared writable filesystem, and two exclusive eight-GPU B200 nodes.
Miles and Slime also require Slurm Pyxis/Enroot support for `--container-image`, root remapping, and writable ephemeral containers.
Their commands request 80 CPU cores per node and expose 64 cores to Ray.
Prime-RL runs from a virtual environment on the hosts and also requests 80 CPU cores per node.
This package does not install Slurm, configure Kubernetes, modify cluster services, or create a container image.

The generation node must appear first in the expanded Slurm node list.
The trainer IPv4 address must sort numerically before the generation address because the recorded Ray placement uses that order.
The site example uses documentation-only addresses. Replace them with routable addresses between your nodes.
The preparation tool rejects unsupported ordering. The GPU preflight independently checks physical placement before loading the models.
The two framework roles use trusted cluster networking; do not expose Ray or model endpoints to the public internet.

The asset directories must be under the configured `assets` path so the containers can access them.
The prepared campaign directory and asset paths must be identical on both nodes.
Paths must not contain spaces, quotes, or shell metacharacters.

## Pinned models

The exact model identities are as follows. Model weights are not stored in this repository.
Use an authenticated Hugging Face client when required. Keep its token outside this repository and outside `site.local.json`.

| Role | Hugging Face repository | Revision |
|---|---|---|
| Base | `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Countdown teacher | `semianalysisai/Qwen3.6-35B-A3B-countdown-GRPO-20260909` | `237a7f0345e883705026acd7f0745a3637042f73` |
| Graph Coloring teacher | `semianalysisai/Qwen3.6-35B-A3B-graph-color-GRPO-20260909` | `ec87f87177a4ced7256928ce67c438fafa73c28e` |

The following commands download the pinned snapshots. They can consume substantial storage and bandwidth. They do not launch training.
Use paths that match your site configuration.

```bash
hf download Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --local-dir /shared/opd-assets/models/Qwen3.6-35B-A3B
hf download semianalysisai/Qwen3.6-35B-A3B-countdown-GRPO-20260909 --revision 237a7f0345e883705026acd7f0745a3637042f73 --local-dir /shared/opd-assets/teachers/countdown
hf download semianalysisai/Qwen3.6-35B-A3B-graph-color-GRPO-20260909 --revision ec87f87177a4ced7256928ce67c438fafa73c28e --local-dir /shared/opd-assets/teachers/graph_color
```

Download the teachers as frozen scoring models. MOPD trains the student, not these teachers.
The data files are included, so do not regenerate a different dataset when reproducing the recorded experiment.

## Prepare a campaign

```bash
cp site.example.json site.local.json
# Edit site.local.json before continuing.
python3 tools/prepare.py slime --site site.local.json --output /shared/opd-runs/slime-01
```

The tool checks out the recorded revision, applies the checked patch, expands the site values, and verifies every dataset hash.
It creates model symlinks and retains `site.json` plus `package-provenance.json` in the new campaign.
The tool performs no model conversion, environment installation, scheduler submission, or automatic retry.
If preparation fails, inspect the partial directory and select a fresh output path for the next attempt.

## Miles and Slime runtime

Both experiments reused a container identified as `miles-dev-202609050049.sqsh`.
The image bytes are not distributed here, and its name alone is not a content digest.
The recorded runtime contained PyTorch 2.13.0+cu130, SGLang 0.5.19.dev56+ga359142, Ray 2.58.0, Transformers 5.12.1,
Transformer Engine 2.17.0, Flash Linear Attention 0.5.2, and SGLang Router 0.3.2.
SGLang's recorded source revision was `a3591427c20431d33570f0c766d0cee6711f5ee8`.
The Megatron source revision was `8c1e05747eb612b382df2632783df5c83a853646`.
Set `megatron_source` to its path inside the container.

Use the original compatible image or provision a compatible runtime separately.
The package does not claim that installing these version strings recreates the original CUDA binaries.
The original image used several local compiled wheels, so its package inventory is evidence rather than a universal installation lock.
The Slime package retains the inventory observed during the shared-image preflight in `recorded-runtime.json`.
The framework-specific preflight must pass in the actual runtime before training.

Both frameworks require a converted Megatron `torch_dist` base checkpoint at `megatron_model`.
The recorded runs reused an existing conversion; no fresh conversion or restore test is claimed for this package.
Use the conversion implementation in the pinned upstream source and validate its configuration against the base model.
Preparation links the configured conversion to `models/Qwen3.6-35B-A3B_torch_dist`.
Model conversion can require GPUs and substantial storage, so it is not silently performed by preparation.

Slime additionally installs NumPy 1.26.4 and SciPy 1.15.3 inside each ephemeral container.
Download the exact CPython 3.12 Linux wheels into the prepared campaign, then verify their recorded digests.

```bash
python3 -m pip download --only-binary=:all: --no-deps --platform manylinux2014_x86_64 --python-version 312 --implementation cp --abi cp312 --dest /shared/opd-runs/slime-01/wheels numpy==1.26.4 scipy==1.15.3
cd /shared/opd-runs/slime-01
sha256sum -c wheels.sha256
```

Slime checks immutable SGLang arguments, weight-update session ordering, native NCCL collectives, asynchronous waits, and the campaign objective before training.
It uses native NCCL groups only with trainer offload and release disabled.
Miles checks candidate scoring, the domain-balanced async buffer, bounded scoring retries, evaluation, and TensorBoard initialization.

## Prime-RL runtime

Prime-RL requires a working environment at the prepared campaign's `source/.venv`.
The original environment used PyTorch 2.13.0+cu130 and vLLM 0.28.0.
Its recorded package inventory is `frameworks/prime-rl/recorded-packages.txt`.
The pinned source includes `uv.lock`, and preparation initializes its submodules at their recorded commits over HTTPS.
Follow the installation instructions in that pinned source. Ensure that the generated task environment is installed into the same environment.

```bash
# Run this only after provisioning source/.venv with the pinned framework dependencies.
/usr/local/bin/uv pip install --python /shared/opd-runs/prime-01/source/.venv/bin/python --no-deps -e /shared/opd-runs/prime-01/environments/mopd_puzzles
```

Use the `uv` executable configured in your site file if its path differs.
The controller deliberately uses `--no-sync`, so submission does not silently replace the runtime.
Prime-RL requires a completed native conversion at `base_model/prime`, including `.prime-v1` and the safetensors index.
The preflight checks all indexed shard headers, lengths, tokenizer agreement, scoring cases, and the frozen-vision attention configuration.
A partial conversion is rejected. Provision and validate this conversion through the pinned Prime-RL implementation before submission.

## Inspect, submit, and monitor

The following command prints the allocation request without executing it.

```bash
python3 tools/submit.py /shared/opd-runs/slime-01 --partition YOUR_PARTITION
```

Only the following form submits work. Run it on the Slurm controller or from a suitable Slurm pod.
For Kubernetes, enter the appropriate Slurm pod using your own context and namespace before running the command.
No particular Kubernetes context, credentials, or pod name is built into this repository.

```bash
python3 tools/submit.py /shared/opd-runs/slime-01 --partition YOUR_PARTITION --submit
```

The submission process remains attached. Use your normal persistent terminal if the login connection can disconnect.
It holds a per-campaign lock and refuses another run when a previous terminal record is absent.
The controller stops its owned processes when training finishes or fails. It does not repair failures or resubmit automatically.
Read the job number from `active-run.json`, use `squeue` and `sacct` for scheduler state, and follow `results/*/learner-train.out`.
Use `scancel YOUR_JOB_ID` to cancel only the job you intend to stop.

Each run retains source, launch commands, metrics, GPU samples, node counters, teacher timing, and component logs.
Slime also retains samples and rollout tensors. Prime-RL retains native compressed traces.
These files can contain full prompts and responses. Review them before sharing.
Raw logs and model checkpoints are excluded from this repository.

## Known limits

The original runs used 16 B200 GPUs. This package does not establish support for a different GPU or node count.
The runtime and conversion requirements above prevent a claim of fresh-machine, one-command reproduction.
The package makes these requirements explicit and retains the scientific recipe rather than silently substituting another environment.
Perform a separately authorized GPU validation before advertising a new image or hardware configuration as equivalent.
