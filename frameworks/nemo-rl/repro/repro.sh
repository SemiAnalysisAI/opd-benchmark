#!/usr/bin/env bash
# Three NeMo-RL v0.7.0 container problems, each checked in about two minutes on one GPU with nothing but
# NVIDIA's public container. The production deadlock is reproduced by `tools/reproduce.sh nemo-rl`.
#
#   ./repro.sh image         container provenance, venv layout, eager-only Qwen3.5 recipes
#   ./repro.sh uv-override   ray.sub's UV_CACHE_DIR_OVERRIDE empties the image's venv
#   ./repro.sh readonly      the image cannot start read-only, the way ray.sub runs it
#
# Settings (environment): PARTITION (required), WORKDIR (shared, default $PWD/nemo-rl-repro),
# CONTAINER (default docker://nvcr.io#nvidia/nemo-rl:v0.7.0; a local .sqsh is faster), NODE (optional).
set -euo pipefail

: "${PARTITION:?Set PARTITION to your Slurm partition}"
WORKDIR=${WORKDIR:-$PWD/nemo-rl-repro}
CONTAINER=${CONTAINER:-docker://nvcr.io#nvidia/nemo-rl:v0.7.0}
TAG_COMMIT=81aa43dda4765b0429cf31dab44441e4e4383911  # git rev-parse v0.7.0
mkdir -p "$WORKDIR"

# Slurm often sets NVIDIA_VISIBLE_DEVICES=void for job steps, which gives Pyxis containers no GPUs.
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

one_node() {  # one_node [extra srun args...] -- command...
  local args=()
  while [[ $1 != -- ]]; do args+=("$1"); shift; done; shift
  srun -p "$PARTITION" -N1 -n1 ${NODE:+-w $NODE} --gres=gpu:1 --time=00:15:00 --container-image="$CONTAINER" \
    --no-container-mount-home "${args[@]}" bash -c "$*"
}

case "${1:-}" in
image)
  one_node --container-writable -- '
    cd /opt/nemo-rl
    echo "EXPECT NEMO_RL_COMMIT = v0.7.0 tag '"$TAG_COMMIT"'"
    echo "SAW    NEMO_RL_COMMIT = $NEMO_RL_COMMIT"
    [ "$NEMO_RL_COMMIT" = '"$TAG_COMMIT"' ] && echo "=> image matches its tag" || echo "=> image is NOT built from its tag"
    f=$(python -c "import transformers, os; print(os.path.join(os.path.dirname(transformers.__file__), \"__init__.py\"))")
    echo "venv file: $f -> $(readlink "$f" || echo "(regular file)")"
    echo "Qwen3.5-35B-A3B recipes and their vLLM enforce_eager:"
    grep -H "enforce_eager" examples/configs/recipes/llm/*qwen3.5-35ba3b* || true
    grep -n "enforce_eager" examples/configs/grpo_math_1B.yaml'
  ;;
uv-override)
  # ray.sub: `MOUNTS+=",$UV_CACHE_DIR_OVERRIDE:/root/.cache/uv"`. Same mount, then import what grpo.py imports.
  mkdir -p "$WORKDIR/uv-cache"
  one_node --container-writable --container-mounts="$WORKDIR/uv-cache:/root/.cache/uv" -- '
    cd /opt/nemo-rl
    echo "EXPECT: from transformers import AutoProcessor  (first import in nemo_rl/algorithms/grpo.py)"
    python -c "from transformers import AutoProcessor; print(\"=> OK\")" 2>&1 | tail -n 1'
  ;;
readonly)
  # ray.sub starts every container without --container-writable (Pyxis default: read-only).
  one_node -- '
    cd /opt/nemo-rl
    echo "EXPECT: uv run python -c \"import nemo_rl\"  (the launch command upstream documents)"
    uv run python -c "import nemo_rl; print(\"=> OK\")" 2>&1 | tail -n 3'
  ;;
*)
  sed -n 2,10p "$0"; exit 2 ;;
esac
