#!/usr/bin/env bash
set -euo pipefail

bench_root=${BENCH_ROOT:-/root/opd-bench}
artifact_root="$bench_root/artifacts/mopd-frameworks"
run_id=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
prompts_per_domain=${PROMPTS_PER_DOMAIN:-8}
rollouts_per_prompt=${ROLLOUTS_PER_PROMPT:-4}
max_new_tokens=${MAX_NEW_TOKENS:-256}
run_dir="$bench_root/mopd/runs/routing-probe-$run_id"
mkdir -p "$run_dir"

student_pid=
math_pid=
code_pid=
sampler_pid=
cleanup() {
  if [[ -n "$sampler_pid" ]]; then kill "$sampler_pid" 2>/dev/null || true; fi
  if [[ -n "$student_pid" ]]; then kill "$student_pid" 2>/dev/null || true; fi
  if [[ -n "$math_pid" ]]; then kill "$math_pid" 2>/dev/null || true; fi
  if [[ -n "$code_pid" ]]; then kill "$code_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local port=$1
  local pid=$2
  local log=$3
  for _ in $(seq 1 300); do
    curl -fsS "http://127.0.0.1:$port/health_generate" >/dev/null 2>&1 && return 0
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -100 "$log"
      return 1
    fi
    sleep 1
  done
  return 1
}

export HF_HOME="$bench_root/cache/huggingface"
export PYTHONUNBUFFERED=1
process_start_epoch=$(date +%s)
"$artifact_root/gpu_sampler.sh" "$run_dir/gpu.csv" &
sampler_pid=$!

CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path "$bench_root/mopd/models/student" \
  --host 0.0.0.0 --port 13140 --tp 1 \
  --chunked-prefill-size 4096 --mem-fraction-static 0.5 \
  --disable-cuda-graph > "$run_dir/student.log" 2>&1 &
student_pid=$!

CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
  --model-path "$bench_root/mopd/models/math-teacher" \
  --host 0.0.0.0 --port 13141 --tp 1 \
  --chunked-prefill-size 4096 --mem-fraction-static 0.38 \
  --disable-cuda-graph > "$run_dir/math-teacher.log" 2>&1 &
math_pid=$!

wait_for_server 13140 "$student_pid" "$run_dir/student.log"
wait_for_server 13141 "$math_pid" "$run_dir/math-teacher.log"

CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
  --model-path "$bench_root/mopd/models/code-teacher" \
  --host 0.0.0.0 --port 13142 --tp 1 \
  --chunked-prefill-size 4096 --mem-fraction-static 0.38 \
  --disable-cuda-graph > "$run_dir/code-teacher.log" 2>&1 &
code_pid=$!
wait_for_server 13142 "$code_pid" "$run_dir/code-teacher.log"
servers_ready_epoch=$(date +%s)

probe_start_epoch=$(date +%s)
python3 "$artifact_root/routing_probe.py" \
  --data "$bench_root/mopd/data/miles_train.jsonl" \
  --student-model "$bench_root/mopd/models/student" \
  --output-dir "$run_dir" \
  --prompts-per-domain "$prompts_per_domain" \
  --rollouts-per-prompt "$rollouts_per_prompt" \
  --max-new-tokens "$max_new_tokens" --temperature 1.0 \
  2>&1 | tee "$run_dir/run.log"
end_epoch=$(date +%s)

printf '%s\n' \
  "run_id=$run_id" "start_epoch=$process_start_epoch" \
  "servers_ready_epoch=$servers_ready_epoch" "probe_start_epoch=$probe_start_epoch" \
  "end_epoch=$end_epoch" \
  "server_start_seconds=$((servers_ready_epoch - process_start_epoch))" \
  "probe_wall_seconds=$((end_epoch - probe_start_epoch))" \
  "wall_seconds=$((end_epoch - process_start_epoch))" > "$run_dir/wall_time.env"

printf '%s\n' "$run_dir"
