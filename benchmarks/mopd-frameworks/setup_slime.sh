#!/usr/bin/env bash
set -euo pipefail

# Prepare the official Slime MOPD draft implementation and the shared workload.
# Slime MOPD is not on main yet, so both the PR head and its image are pinned.
bench_root=${BENCH_ROOT:-/root/opd-bench}
artifact_root=${ARTIFACT_ROOT:-$bench_root/artifacts/mopd-frameworks}
source_root="$bench_root/src"
model_root="$bench_root/mopd/models"
data_root="$bench_root/mopd/data"
slime_dir="$source_root/slime"
slime_commit=f51403558a47290d190fc9dfabe1859be73aca4f
# Digest for the PR's docker/version.txt tag: nightly-dev-20260608b (linux/amd64).
slime_image=${SLIME_IMAGE:-slimerl/slime@sha256:8f38e20a89cf4e48579b54d04139ce34636713a224d09741af48f1860dc959bd}

student_revision=7ae557604adf67be50417f59c2c2f167def9a775
math_revision=aafeb0fc6f22cbf0eaeed126eff8be45b0360a35
code_revision=2e1fd397ee46e1388853d2af2c993145b0f1098a

mkdir -p "$source_root" "$model_root" "$data_root" "$bench_root/bin"

if [[ ! -d "$slime_dir/.git" ]]; then
  git clone https://github.com/THUDM/slime.git "$slime_dir"
elif [[ -n $(git -C "$slime_dir" status --porcelain) ]]; then
  printf 'Refusing to replace local changes in %s\n' "$slime_dir" >&2
  exit 1
fi
git -C "$slime_dir" fetch origin refs/pull/2051/head
if ! git -C "$slime_dir" cat-file -e "$slime_commit^{commit}"; then
  git -C "$slime_dir" fetch origin "$slime_commit"
fi
git -C "$slime_dir" checkout --detach "$slime_commit"

if [[ ! -x "$bench_root/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$bench_root/bin" sh
fi

download_model() {
  local repo=$1 revision=$2 destination=$3
  if [[ -f "$destination/config.json" && -f "$destination/tokenizer.json" ]]; then
    return
  fi
  "$bench_root/bin/uvx" --from 'huggingface_hub[cli]' hf download "$repo" \
    --revision "$revision" --local-dir "$destination"
}
download_model Qwen/Qwen2.5-0.5B-Instruct "$student_revision" "$model_root/student"
download_model Qwen/Qwen2.5-Math-1.5B-Instruct "$math_revision" "$model_root/math-teacher"
download_model Qwen/Qwen2.5-Coder-1.5B-Instruct "$code_revision" "$model_root/code-teacher"

"$bench_root/bin/uv" run --with datasets --with pyarrow \
  "$artifact_root/prepare_data.py" --output-dir "$data_root"
"$bench_root/bin/uv" run "$artifact_root/prepare_slime_data.py" \
  --input "$data_root/miles_train.jsonl" --output "$data_root/slime_train.jsonl"
"$bench_root/bin/uv" run "$artifact_root/validate_slime_recipe.py" \
  --data-dir "$data_root" --model-dir "$model_root" --slime-dir "$slime_dir"

docker pull "$slime_image"
if docker inspect opd-slime >/dev/null 2>&1; then
  existing_image=$(docker inspect --format '{{.Config.Image}}' opd-slime)
  if [[ "$existing_image" != "$slime_image" ]]; then
    printf 'Container opd-slime uses %s; expected %s. Remove or rename it explicitly.\n' \
      "$existing_image" "$slime_image" >&2
    exit 1
  fi
else
  docker run -d --gpus all --network host --ipc host --shm-size 64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --name opd-slime -v "$bench_root:$bench_root" \
    -v "$slime_dir:/workspace/slime" -w /workspace/slime \
    "$slime_image" sleep infinity
fi
docker start opd-slime >/dev/null

docker exec opd-slime bash -lc 'cd /workspace/slime && pip install -e . --no-deps'

if [[ ! -f "$model_root/student_torch_dist/.slime-conversion-complete" ]]; then
  docker exec -e BENCH_ROOT="$bench_root" opd-slime bash -lc '
    set -euo pipefail
    cd /workspace/slime
    source scripts/models/qwen2.5-0.5B.sh
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "$BENCH_ROOT/mopd/models/student" \
      --save "$BENCH_ROOT/mopd/models/student_torch_dist"
    touch "$BENCH_ROOT/mopd/models/student_torch_dist/.slime-conversion-complete"
  '
fi

printf '%s\n' \
  "Slime MOPD setup complete." \
  "source_commit=$slime_commit" \
  "container_image=$slime_image" \
  "Run: BENCH_ROOT=$bench_root ARTIFACT_ROOT=$artifact_root bash $artifact_root/run_slime.sh"
