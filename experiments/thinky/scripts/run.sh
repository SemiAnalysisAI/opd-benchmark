#!/usr/bin/env bash
set -euo pipefail

if [[ -z ${TINKER_API_KEY:-} ]]; then
  printf 'TINKER_API_KEY is required; create one in the Tinker console.\n' >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
experiment_root=$(cd "$script_dir/.." && pwd)
repo_root=$(cd "$experiment_root/../.." && pwd)
run_id=${RUN_ID:-tinker-$(date -u +%Y%m%d-%H%M%S)}
result_root=${RESULT_ROOT:-$experiment_root/results}
result_dir=${RESULT_DIR:-$result_root/$run_id}
console_log="$result_root/$run_id.console.log"
python_bin=${PYTHON:-python3}

mkdir -p "$result_root"
if [[ -e "$result_dir" ]]; then
  printf 'Result directory already exists: %s\n' "$result_dir" >&2
  exit 2
fi

args=(
  --student-model "${STUDENT_MODEL:-Qwen/Qwen3.5-4B}"
  --teacher-model "${TEACHER_MODEL:-Qwen/Qwen3.5-9B}"
  --renderer-name "${RENDERER_NAME:-qwen3_5_disable_thinking}"
  --steps "${STEPS:-15}"
  --prompts-per-step "${PROMPTS_PER_STEP:-8}"
  --rollouts-per-prompt "${ROLLOUTS_PER_PROMPT:-16}"
  --max-prompt-tokens "${MAX_PROMPT_TOKENS:-512}"
  --max-tokens "${MAX_TOKENS:-128}"
  --temperature "${TEMPERATURE:-1.0}"
  --learning-rate "${LEARNING_RATE:-3e-6}"
  --kl-penalty-coef "${KL_PENALTY_COEF:-1.0}"
  --lora-rank "${LORA_RANK:-32}"
  --eval-size "${EVAL_SIZE:-128}"
  --eval-every "${EVAL_EVERY:-0}"
  --save-every "${SAVE_EVERY:-15}"
  --log-path "$result_dir"
)
if [[ -n ${STUDENT_CHECKPOINT:-} ]]; then args+=(--student-checkpoint "$STUDENT_CHECKPOINT"); fi
if [[ -n ${TEACHER_CHECKPOINT:-} ]]; then args+=(--teacher-checkpoint "$TEACHER_CHECKPOINT"); fi
if [[ -n ${WANDB_PROJECT:-} ]]; then args+=(--wandb-project "$WANDB_PROJECT"); fi
if [[ -n ${WANDB_NAME:-} ]]; then args+=(--wandb-name "$WANDB_NAME"); fi

start_epoch=$(date +%s)
set +e
"$python_bin" "$script_dir/run_opd.py" "${args[@]}" "$@" 2>&1 | tee "$console_log"
status=${PIPESTATUS[0]}
set -e
end_epoch=$(date +%s)

saved_console_log=$console_log
if [[ -d "$result_dir" ]]; then
  mv "$console_log" "$result_dir/run.log"
  saved_console_log="$result_dir/run.log"
  printf '%s\n' \
    'provider=tinker' \
    "run_id=$run_id" \
    "start_epoch=$start_epoch" \
    "end_epoch=$end_epoch" \
    "wall_seconds=$((end_epoch - start_epoch))" \
    > "$result_dir/wall_time.env"
fi
if [[ "$status" -ne 0 ]]; then
  printf 'Tinker run failed with status %s. Console log: %s\n' "$status" "$saved_console_log" >&2
  exit "$status"
fi

"$python_bin" "$script_dir/analyze.py" \
  --tinker-results "$result_dir" \
  --local-summary "${LOCAL_SUMMARY:-$repo_root/benchmarks/opd-frameworks/results/summary.json}"

printf '%s\n' "$result_dir"
