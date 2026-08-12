#!/usr/bin/env bash
set -euo pipefail

bench_root=${BENCH_ROOT:-/root/opd-bench}
run_id=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
steps=${STEPS:-15}
run_dir="$bench_root/runs/miles-$run_id"
mkdir -p "$run_dir"

teacher_pid=
sampler_pid=
cleanup() {
  if [[ -n "$sampler_pid" ]]; then kill "$sampler_pid" 2>/dev/null || true; fi
  if [[ -n "$teacher_pid" ]]; then kill "$teacher_pid" 2>/dev/null || true; fi
  CUDA_VISIBLE_DEVICES=0 ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cd /workspace/miles
export HF_HOME="$bench_root/cache/huggingface"
export PYTHONUNBUFFERED=1

process_start_epoch=$(date +%s)
"$bench_root/artifacts/opd-frameworks/gpu_sampler.sh" "$run_dir/gpu.csv" &
sampler_pid=$!

CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
  --model-path "$bench_root/models/teacher" \
  --host 0.0.0.0 \
  --port 13141 \
  --tp 1 \
  --chunked-prefill-size 4096 \
  --mem-fraction-static 0.5 \
  --disable-cuda-graph \
  > "$run_dir/teacher.log" 2>&1 &
teacher_pid=$!

for _ in $(seq 1 300); do
  if curl -fsS http://127.0.0.1:13141/health_generate >/dev/null 2>&1; then break; fi
  if ! kill -0 "$teacher_pid" 2>/dev/null; then
    tail -100 "$run_dir/teacher.log"
    exit 1
  fi
  sleep 1
done
curl -fsS http://127.0.0.1:13141/health_generate >/dev/null 2>&1
teacher_ready_epoch=$(date +%s)

export CUDA_VISIBLE_DEVICES=0
ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 > "$run_dir/ray.log" 2>&1
ray_ready_epoch=$(date +%s)

source scripts/models/qwen3-0.6B.sh
training_start_epoch=$(date +%s)
ray job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"/root/Megatron-LM/:$bench_root/artifacts/opd-frameworks\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"}}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  --colocate \
  --hf-checkpoint "$bench_root/models/student" \
  --ref-load "$bench_root/models/student_torch_dist" \
  --prompt-data "$bench_root/data/reverse-text/train.jsonl" \
  --input-key messages \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rollout-all-samples-process-path miles_task_score.process_samples \
  --num-rollout "$steps" \
  --rollout-batch-size 8 \
  --n-samples-per-prompt 16 \
  --rollout-max-response-len 128 \
  --rollout-temperature 1.0 \
  --rollout-top-p 1.0 \
  --seed 42 \
  --global-batch-size 128 \
  --balance-data \
  --custom-rm-path miles.rollout.on_policy_distillation.reward_func \
  --custom-reward-post-process-path miles.rollout.on_policy_distillation.post_process_rewards \
  --rm-url http://127.0.0.1:13141/generate \
  --advantage-estimator grpo \
  --use-opd \
  --opd-type sglang \
  --opd-kl-coef 1.0 \
  --opd-log-prob-top-k 0 \
  --use-kl-loss \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 3e-6 \
  --lr-decay-style constant \
  --weight-decay 0.01 \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --clip-grad 1.0 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --sequence-parallel \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 8192 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.5 \
  --sglang-enable-metrics \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --bf16 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  "${MODEL_ARGS[@]}" \
  2>&1 | tee "$run_dir/run.log"
end_epoch=$(date +%s)

printf '%s\n' \
  "framework=miles" \
  "run_id=$run_id" \
  "start_epoch=$process_start_epoch" \
  "teacher_ready_epoch=$teacher_ready_epoch" \
  "ray_ready_epoch=$ray_ready_epoch" \
  "training_start_epoch=$training_start_epoch" \
  "end_epoch=$end_epoch" \
  "teacher_start_seconds=$((teacher_ready_epoch - process_start_epoch))" \
  "ray_start_seconds=$((ray_ready_epoch - teacher_ready_epoch))" \
  "framework_wall_seconds=$((end_epoch - training_start_epoch))" \
  "wall_seconds=$((end_epoch - process_start_epoch))" \
  > "$run_dir/wall_time.env"

printf '%s\n' "$run_dir"
