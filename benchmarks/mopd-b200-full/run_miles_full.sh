#!/usr/bin/env bash
set -euo pipefail

bench_root=${BENCH_ROOT:-/dev/shm/opd-bench}
artifact_root=${ARTIFACT_ROOT:-$bench_root/artifacts/mopd-b200-full}
run_id=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
steps=${STEPS:-15}
train_attention_backend=${TRAIN_ATTENTION_BACKEND:-fused}
run_dir="$bench_root/runs-full/miles-$run_id"
mkdir -p "$run_dir"

math_pid=
code_pid=
sampler_pid=
cleanup() {
  if [[ -n "$sampler_pid" ]]; then kill "$sampler_pid" 2>/dev/null || true; fi
  if [[ -n "$math_pid" ]]; then kill "$math_pid" 2>/dev/null || true; fi
  if [[ -n "$code_pid" ]]; then kill "$code_pid" 2>/dev/null || true; fi
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local port=$1 pid=$2 log=$3
  for _ in $(seq 1 600); do
    curl -fsS "http://127.0.0.1:$port/health_generate" >/dev/null 2>&1 && return 0
    if ! kill -0 "$pid" 2>/dev/null; then tail -120 "$log"; return 1; fi
    sleep 1
  done
  tail -120 "$log"
  return 1
}

cd /workspace/miles
export HF_HOME="$bench_root/cache/huggingface"
export PYTHONUNBUFFERED=1
process_start_epoch=$(date +%s)
"$artifact_root/gpu_sampler.sh" "$run_dir/gpu.csv" &
sampler_pid=$!

CUDA_VISIBLE_DEVICES=6 python3 -m sglang.launch_server \
  --model-path "$bench_root/models-full/math-teacher" --host 0.0.0.0 --port 13141 \
  --tp 1 --chunked-prefill-size 4096 --mem-fraction-static 0.70 \
  > "$run_dir/math-teacher.log" 2>&1 &
math_pid=$!
CUDA_VISIBLE_DEVICES=7 python3 -m sglang.launch_server \
  --model-path "$bench_root/models-full/code-teacher" --host 0.0.0.0 --port 13142 \
  --tp 1 --chunked-prefill-size 4096 --mem-fraction-static 0.70 \
  > "$run_dir/code-teacher.log" 2>&1 &
code_pid=$!
wait_for_server 13141 "$math_pid" "$run_dir/math-teacher.log"
wait_for_server 13142 "$code_pid" "$run_dir/code-teacher.log"
teachers_ready_epoch=$(date +%s)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus 6 --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port=8265 > "$run_dir/ray.log" 2>&1
ray_ready_epoch=$(date +%s)

source scripts/models/qwen3-8B.sh
training_start_epoch=$(date +%s)
PYTHONPATH=/root/Megatron-LM CUDA_DEVICE_MAX_CONNECTIONS=1 python3 train.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 4 \
  --hf-checkpoint "$bench_root/models-full/student" \
  --ref-load "$bench_root/models-full/student_torch_dist" \
  --prompt-data "$bench_root/data-full/miles_train.jsonl" \
  --input-key messages --label-key label --apply-chat-template --rollout-shuffle \
  --num-rollout "$steps" --rollout-batch-size 16 --n-samples-per-prompt 4 \
  --rollout-max-response-len 16384 --rollout-temperature 1.0 --rollout-top-p 1.0 \
  --seed 42 --global-batch-size 64 --balance-data \
  --custom-rm-path miles.rollout.on_policy_distillation.reward_func \
  --custom-reward-post-process-path miles.rollout.on_policy_distillation.post_process_rewards \
  --opd-teacher-urls math=http://127.0.0.1:13141/generate code=http://127.0.0.1:13142/generate \
  --opd-teacher-key opd_teacher --advantage-estimator grpo --use-opd --opd-type sglang \
  --opd-kl-coef 1.0 --opd-log-prob-top-k 0 \
  --use-kl-loss --kl-loss-coef 0.0 --kl-loss-type low_var_kl \
  --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --clip-grad 1.0 \
  --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 --sequence-parallel \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 16384 \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.70 \
  --sglang-enable-metrics --attention-dropout 0.0 --hidden-dropout 0.0 \
  --bf16 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend "$train_attention_backend" "${MODEL_ARGS[@]}" \
  2>&1 | tee "$run_dir/run.log"
end_epoch=$(date +%s)

printf '%s\n' \
  "framework=miles" "run_id=$run_id" "start_epoch=$process_start_epoch" \
  "train_attention_backend=$train_attention_backend" \
  "teachers_ready_epoch=$teachers_ready_epoch" "ray_ready_epoch=$ray_ready_epoch" \
  "training_start_epoch=$training_start_epoch" "end_epoch=$end_epoch" \
  "teacher_start_seconds=$((teachers_ready_epoch-process_start_epoch))" \
  "framework_wall_seconds=$((end_epoch-training_start_epoch))" \
  "wall_seconds=$((end_epoch-process_start_epoch))" > "$run_dir/wall_time.env"
printf '%s\n' "$run_dir"
