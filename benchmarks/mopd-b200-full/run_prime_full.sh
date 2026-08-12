#!/usr/bin/env bash
set -euo pipefail

bench_root=${BENCH_ROOT:-/dev/shm/opd-bench}
artifact_root=${ARTIFACT_ROOT:-$bench_root/artifacts/mopd-b200-full}
source_root="$bench_root/src/prime-rl"
run_id=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
steps=${STEPS:-15}
run_dir="$bench_root/runs-full/prime-$run_id"
mkdir -p "$run_dir"

math_pid=
code_pid=
sampler_pid=
cleanup() {
  if [[ -n "$sampler_pid" ]]; then kill "$sampler_pid" 2>/dev/null || true; fi
  if [[ -n "$math_pid" ]]; then kill "$math_pid" 2>/dev/null || true; fi
  if [[ -n "$code_pid" ]]; then kill "$code_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local url=$1 pid=$2 log=$3
  for _ in $(seq 1 600); do
    curl -fsS "$url" >/dev/null 2>&1 && return 0
    if ! kill -0 "$pid" 2>/dev/null; then tail -120 "$log"; return 1; fi
    sleep 1
  done
  tail -120 "$log"
  return 1
}

cd "$source_root"
export HF_HOME="$bench_root/cache/huggingface"
export WANDB_MODE=disabled
export PATH="$source_root/.venv/bin:$bench_root/bin:$PATH"
process_start_epoch=$(date +%s)
"$artifact_root/gpu_sampler.sh" "$run_dir/gpu.csv" &
sampler_pid=$!

CUDA_VISIBLE_DEVICES=6 .venv/bin/inference \
  --model.name "$bench_root/models-full/math-teacher" \
  --server.port 8001 --gpu-memory-utilization 0.75 --seed 42 \
  --model.max-model-len 17408 --model.enforce-eager \
  > "$run_dir/math-teacher.log" 2>&1 &
math_pid=$!
CUDA_VISIBLE_DEVICES=7 .venv/bin/inference \
  --model.name "$bench_root/models-full/code-teacher" \
  --server.port 8002 --gpu-memory-utilization 0.75 --seed 42 \
  --model.max-model-len 17408 --model.enforce-eager \
  > "$run_dir/code-teacher.log" 2>&1 &
code_pid=$!
wait_for_server http://127.0.0.1:8001/v1/models "$math_pid" "$run_dir/math-teacher.log"
wait_for_server http://127.0.0.1:8002/v1/models "$code_pid" "$run_dir/code-teacher.log"
teachers_ready_epoch=$(date +%s)

training_start_epoch=$(date +%s)
# Prime allocates inference GPUs before trainer GPUs. This ordering maps its
# four rollout GPUs to physical 2-5 and its two trainer GPUs to physical 0-1.
CUDA_VISIBLE_DEVICES=2,3,4,5,0,1 .venv/bin/rl \
  @ "$artifact_root/prime_full.toml" \
  --max-steps "$steps" --output-dir "$run_dir/output" --clean-output-dir \
  > "$run_dir/run.log" 2>&1
end_epoch=$(date +%s)

printf '%s\n' \
  "framework=prime-rl" "run_id=$run_id" "start_epoch=$process_start_epoch" \
  "teachers_ready_epoch=$teachers_ready_epoch" "training_start_epoch=$training_start_epoch" \
  "end_epoch=$end_epoch" "teacher_start_seconds=$((teachers_ready_epoch-process_start_epoch))" \
  "framework_wall_seconds=$((end_epoch-training_start_epoch))" \
  "wall_seconds=$((end_epoch-process_start_epoch))" > "$run_dir/wall_time.env"
printf '%s\n' "$run_dir"
