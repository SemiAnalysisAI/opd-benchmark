#!/usr/bin/env bash
set -euo pipefail

bench_root=${BENCH_ROOT:-/root/opd-bench}
source_root="$bench_root/src/prime-rl"
artifact_root="$bench_root/artifacts/mopd-frameworks"
run_id=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
steps=${STEPS:-15}
teacher_mode=${TEACHER_MODE:-multi}
run_dir="$bench_root/mopd/runs/prime-$run_id"
mkdir -p "$run_dir"

case "$teacher_mode" in
  single) config_file="$artifact_root/prime_opd.toml" ;;
  multi) config_file="$artifact_root/prime.toml" ;;
  *) printf 'TEACHER_MODE must be single or multi, got %s\n' "$teacher_mode" >&2; exit 2 ;;
esac

math_pid=
code_pid=
sampler_pid=
cleanup() {
  if [[ -n "$sampler_pid" ]]; then kill "$sampler_pid" 2>/dev/null || true; fi
  if [[ -n "$math_pid" ]]; then kill "$math_pid" 2>/dev/null || true; fi
  if [[ -n "$code_pid" ]]; then kill "$code_pid" 2>/dev/null || true; fi
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
  --model.name "$bench_root/mopd/models/math-teacher" \
  --server.port 8001 \
  --gpu-memory-utilization 0.24 \
  --seed 42 \
  --model.enforce-eager \
  > "$run_dir/math-teacher.log" 2>&1 &
math_pid=$!

for _ in $(seq 1 300); do
  curl -fsS http://localhost:8001/v1/models >/dev/null 2>&1 && break
  if ! kill -0 "$math_pid" 2>/dev/null; then tail -100 "$run_dir/math-teacher.log"; exit 1; fi
  sleep 1
done
curl -fsS http://localhost:8001/v1/models >/dev/null

if [[ "$teacher_mode" == multi ]]; then
  CUDA_VISIBLE_DEVICES=0 .venv/bin/inference \
    --model.name "$bench_root/mopd/models/code-teacher" \
    --server.port 8002 \
    --gpu-memory-utilization 0.24 \
    --seed 42 \
    --model.enforce-eager \
    > "$run_dir/code-teacher.log" 2>&1 &
  code_pid=$!

  for _ in $(seq 1 300); do
    curl -fsS http://localhost:8002/v1/models >/dev/null 2>&1 && break
    if ! kill -0 "$code_pid" 2>/dev/null; then tail -100 "$run_dir/code-teacher.log"; exit 1; fi
    sleep 1
  done
  curl -fsS http://localhost:8002/v1/models >/dev/null
fi
teachers_ready_epoch=$(date +%s)

training_start_epoch=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/rl \
  @ "$config_file" \
  --max-steps "$steps" \
  --output-dir "$run_dir/output" \
  --clean-output-dir \
  > "$run_dir/run.log" 2>&1
end_epoch=$(date +%s)

printf '%s\n' \
  "framework=prime-rl" \
  "teacher_mode=$teacher_mode" \
  "run_id=$run_id" \
  "start_epoch=$process_start_epoch" \
  "teachers_ready_epoch=$teachers_ready_epoch" \
  "training_start_epoch=$training_start_epoch" \
  "end_epoch=$end_epoch" \
  "teacher_start_seconds=$((teachers_ready_epoch - process_start_epoch))" \
  "framework_wall_seconds=$((end_epoch - training_start_epoch))" \
  "wall_seconds=$((end_epoch - process_start_epoch))" \
  > "$run_dir/wall_time.env"

printf '%s\n' "$run_dir"
