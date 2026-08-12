#!/usr/bin/env bash
set -euo pipefail

bench_root=${BENCH_ROOT:-/root/opd-bench}
source_root="$bench_root/src/prime-rl"
artifact_root="$bench_root/artifacts/opd-frameworks"
run_id=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
steps=${STEPS:-15}
run_dir="$bench_root/runs/prime-$run_id"
mkdir -p "$run_dir" "$artifact_root"

teacher_pid=
sampler_pid=
cleanup() {
  if [[ -n "$sampler_pid" ]]; then kill "$sampler_pid" 2>/dev/null || true; fi
  if [[ -n "$teacher_pid" ]]; then kill "$teacher_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

cd "$source_root"
export HF_HOME="$bench_root/cache/huggingface"
export WANDB_MODE=disabled
export PATH="$source_root/.venv/bin:$bench_root/bin:$PATH"

process_start_epoch=$(date +%s)
"$artifact_root/gpu_sampler.sh" "$run_dir/gpu.csv" &
sampler_pid=$!

CUDA_VISIBLE_DEVICES=0 .venv/bin/inference \
  --model.name "$bench_root/models/teacher" \
  --server.port 8001 \
  --gpu-memory-utilization 0.4 \
  --seed 42 \
  --model.enforce-eager \
  > "$run_dir/teacher.log" 2>&1 &
teacher_pid=$!

for _ in $(seq 1 180); do
  if curl -fsS http://localhost:8001/v1/models >/dev/null 2>&1; then break; fi
  if ! kill -0 "$teacher_pid" 2>/dev/null; then
    tail -100 "$run_dir/teacher.log"
    exit 1
  fi
  sleep 1
done
curl -fsS http://localhost:8001/v1/models >/dev/null 2>&1
teacher_ready_epoch=$(date +%s)

training_start_epoch=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/rl \
  @ "$artifact_root/prime.toml" \
  --max-steps "$steps" \
  --output-dir "$run_dir/output" \
  --clean-output-dir \
  > "$run_dir/run.log" 2>&1
end_epoch=$(date +%s)

printf '%s\n' \
  "framework=prime-rl" \
  "run_id=$run_id" \
  "start_epoch=$process_start_epoch" \
  "teacher_ready_epoch=$teacher_ready_epoch" \
  "training_start_epoch=$training_start_epoch" \
  "end_epoch=$end_epoch" \
  "teacher_start_seconds=$((teacher_ready_epoch - process_start_epoch))" \
  "framework_wall_seconds=$((end_epoch - training_start_epoch))" \
  "wall_seconds=$((end_epoch - process_start_epoch))" \
  > "$run_dir/wall_time.env"

printf '%s\n' "$run_dir"
