#!/usr/bin/env bash
set -euo pipefail

# Recreate the exact source/model/data environment used for REPORT.md.
# Run on a two-H100 Linux host after copying this directory to ARTIFACT_ROOT.
bench_root=${BENCH_ROOT:-/root/opd-bench}
artifact_root=${ARTIFACT_ROOT:-$bench_root/artifacts/mopd-frameworks}
source_root="$bench_root/src"
model_root="$bench_root/mopd/models"
data_root="$bench_root/mopd/data"
mkdir -p "$source_root" "$model_root" "$data_root" "$bench_root/bin"

prime_commit=ec92686fbceb9375d2155cd05c6e87652bf68441
miles_commit=862ac1ea1fba864171b006d45e0d5e92ff008c6a
verl_commit=2b0fe5158be73b30e749f5c63c1c3b6d5db5d614
miles_image=radixark/miles@sha256:37b5eac955caa2104690ac8b55ee2d70579d962546f61b208c6a0392d1df15a6
verl_image=verlai/verl@sha256:b867883b0dd011363e69ab2ab344922a28c5bd0409e2a324e3ee70fb27ca7543

clone_at() {
  local name=$1 url=$2 commit=$3
  if [[ ! -d "$source_root/$name/.git" ]]; then
    git clone "$url" "$source_root/$name"
  fi
  git -C "$source_root/$name" fetch origin "$commit"
  git -C "$source_root/$name" checkout --detach "$commit"
}

clone_at prime-rl https://github.com/PrimeIntellect-ai/prime-rl.git "$prime_commit"
clone_at miles https://github.com/radixark/miles.git "$miles_commit"
clone_at verl https://github.com/verl-project/verl.git "$verl_commit"
git -C "$source_root/prime-rl" submodule update --init --recursive

if [[ ! -x "$bench_root/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$bench_root/bin" sh
fi
cd "$source_root/prime-rl"
"$bench_root/bin/uv" sync --all-extras
"$bench_root/bin/uv" pip install --python .venv/bin/python -e "$artifact_root/prime_mopd_v1"

download_model() {
  local repo=$1 revision=$2 destination=$3
  "$bench_root/bin/uvx" --from 'huggingface_hub[cli]' hf download "$repo" \
    --revision "$revision" --local-dir "$destination"
}
download_model Qwen/Qwen2.5-0.5B-Instruct \
  7ae557604adf67be50417f59c2c2f167def9a775 "$model_root/student"
download_model Qwen/Qwen2.5-Math-1.5B-Instruct \
  aafeb0fc6f22cbf0eaeed126eff8be45b0360a35 "$model_root/math-teacher"
download_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  2e1fd397ee46e1388853d2af2c993145b0f1098a "$model_root/code-teacher"

"$bench_root/bin/uv" run --with datasets --with pyarrow \
  "$artifact_root/prepare_data.py" --output-dir "$data_root"
"$bench_root/bin/uv" run --with datasets --with pyarrow \
  "$artifact_root/validate_recipe.py" --data-dir "$data_root" --model-dir "$model_root"

docker pull "$miles_image"
docker pull "$verl_image"
if ! docker inspect opd-miles >/dev/null 2>&1; then
  docker run -d --gpus all --network host --ipc host --shm-size 64g \
    --name opd-miles -v "$bench_root:$bench_root" \
    -v "$source_root/miles:/workspace/miles" -w /workspace/miles \
    "$miles_image" sleep infinity
fi
if ! docker inspect opd-verl >/dev/null 2>&1; then
  docker run -d --gpus all --network host --ipc host --shm-size 64g \
    --name opd-verl -v "$bench_root:$bench_root" \
    -v "$source_root/verl:/workspace/verl" -w /workspace/verl \
    "$verl_image" sleep infinity
fi

docker exec -e BENCH_ROOT="$bench_root" opd-miles bash -lc '
  set -euo pipefail
  cd /workspace/miles
  source scripts/models/qwen2.5-0.5B.sh
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$BENCH_ROOT/mopd/models/student" \
    --save "$BENCH_ROOT/mopd/models/student_torch_dist"
'

printf '%s\n' "Setup complete. Run run_prime.sh, run_miles.sh, then run_verl.sh."
