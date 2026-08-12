#!/usr/bin/env bash
set -Eeuo pipefail

# Clean-node setup for the full Miles Qwen3-8B MOPD recipe on one 8-GPU
# Blackwell node. Copy this artifact directory to ARTIFACT_ROOT before running.
bench_root=${BENCH_ROOT:-/dev/shm/opd-bench}
artifact_root=${ARTIFACT_ROOT:-$bench_root/artifacts/mopd-b200-full}
source_root="$bench_root/src"
model_root="$bench_root/models-full"
data_root="$bench_root/data-full"
log_root="$bench_root/logs"
mkdir -p "$source_root" "$model_root" "$data_root" "$log_root" "$bench_root/bin"
setup_log="$log_root/full-setup-$(date -u +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$setup_log") 2>&1

prime_commit=ec92686fbceb9375d2155cd05c6e87652bf68441
miles_commit=862ac1ea1fba864171b006d45e0d5e92ff008c6a
verl_commit=2b0fe5158be73b30e749f5c63c1c3b6d5db5d614
miles_image=radixark/miles@sha256:37b5eac955caa2104690ac8b55ee2d70579d962546f61b208c6a0392d1df15a6
verl_image=verlai/verl@sha256:b867883b0dd011363e69ab2ab344922a28c5bd0409e2a324e3ee70fb27ca7543
student_revision=b968826d9c46dd6066d109eabc6255188de91218
math_revision=9216db5781bf21249d130ec9da846c4624c16137
code_revision=b2cff646eb4bb1d68355c01b18ae02e7cf42d120

if [[ ! -f "$artifact_root/run_all_full.sh" ]]; then
  printf 'Missing artifact directory: %s\n' "$artifact_root" >&2
  exit 2
fi

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$gpu_count" -ne 8 ]]; then
  printf 'Expected exactly 8 GPUs, found %s\n' "$gpu_count" >&2
  exit 2
fi
min_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
if (( min_memory < 180000 )); then
  printf 'Expected roughly 192 GB-class GPUs, smallest device reports %s MiB\n' "$min_memory" >&2
  exit 2
fi
available_kib=$(df --output=avail "$bench_root" | tail -1)
if (( available_kib < 180000000 )); then
  printf 'Need at least 180 GB free under %s; df reports %s KiB\n' "$bench_root" "$available_kib" >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential ca-certificates curl git jq python3.12-dev

if [[ ! -x "$bench_root/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$bench_root/bin" sh
fi
export PATH="$bench_root/bin:$PATH"

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

# Prime pins some submodules with SSH URLs. The command-scoped rewrite avoids
# requiring a GitHub SSH key on a clean benchmark node. Restore first so an
# interrupted submodule checkout cannot leave an empty staged directory.
git -C "$source_root/prime-rl" restore --source=HEAD --staged --worktree -- .
git -C "$source_root/prime-rl" submodule sync --recursive
git -C "$source_root/prime-rl" \
  -c 'url.https://github.com/.insteadOf=git@github.com:' \
  -c 'url.https://github.com/.insteadOf=ssh://git@github.com/' \
  submodule update --init --recursive

cd "$source_root/prime-rl"
uv sync --all-extras
uv pip install --python .venv/bin/python -e "$artifact_root/prime_mopd_v1"

download_model() {
  local repo=$1 revision=$2 destination=$3 log_name=$4
  uvx --from 'huggingface_hub[cli]' hf download "$repo" \
    --revision "$revision" --local-dir "$destination" \
    > "$log_root/$log_name" 2>&1
}

download_model Qwen/Qwen3-8B "$student_revision" \
  "$model_root/student" student-download.log &
student_pid=$!
download_model Qwen/Qwen3-32B "$math_revision" \
  "$model_root/math-teacher" math-download.log &
math_pid=$!
download_model Qwen/Qwen3-Coder-30B-A3B-Instruct "$code_revision" \
  "$model_root/code-teacher" code-download.log &
code_pid=$!
wait "$student_pid"
wait "$math_pid"
wait "$code_pid"

uv run --with datasets --with pyarrow \
  "$artifact_root/prepare_data.py" --output-dir "$data_root" \
  | tee "$log_root/data-setup.log"
uv run --with datasets --with pyarrow --with transformers \
  "$artifact_root/validate_full_recipe.py" \
  --data-dir "$data_root" --model-dir "$model_root" \
  | tee "$log_root/recipe-validation.log"

docker pull "$miles_image"
docker pull "$verl_image"
docker rm -f opd-miles-full opd-verl-full >/dev/null 2>&1 || true
docker run -d --gpus all --network host --ipc host --shm-size 128g \
  --name opd-miles-full -v "$bench_root:$bench_root" \
  -v "$source_root/miles:/workspace/miles" -w /workspace/miles \
  "$miles_image" sleep infinity
docker run -d --gpus all --network host --ipc host --shm-size 128g \
  --name opd-verl-full -v "$bench_root:$bench_root" \
  -v "$source_root/verl:/workspace/verl" -w /workspace/verl \
  "$verl_image" sleep infinity

# The image's unpinned Accelerate 1.12.0 forwards a Transformers 5.x internal
# field into torch.nn.Parameter and fails Qwen3 model construction. 1.14.0
# explicitly handles that field.
docker exec opd-verl-full python3 -m pip install --no-cache-dir accelerate==1.14.0 \
  | tee "$log_root/verl-accelerate-upgrade.log"

if [[ ! -f "$model_root/student_torch_dist/.conversion-complete" ]]; then
  docker exec -e BENCH_ROOT="$bench_root" opd-miles-full bash -lc '
    set -euo pipefail
    cd /workspace/miles
    source scripts/models/qwen3-8B.sh
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "$BENCH_ROOT/models-full/student" \
      --save "$BENCH_ROOT/models-full/student_torch_dist"
  ' | tee "$log_root/miles-convert.log"
  touch "$model_root/student_torch_dist/.conversion-complete"
fi

bash -n "$artifact_root/run_prime_full.sh"
bash -n "$artifact_root/run_miles_full.sh"
bash -n "$artifact_root/run_verl_full.sh"
bash -n "$artifact_root/run_all_full.sh"

printf 'Setup complete. Full log: %s\n' "$setup_log"
printf 'Run: STEPS=15 RUN_ID=b200-full BENCH_ROOT=%s ARTIFACT_ROOT=%s bash %s/run_all_full.sh\n' \
  "$bench_root" "$artifact_root" "$artifact_root"
