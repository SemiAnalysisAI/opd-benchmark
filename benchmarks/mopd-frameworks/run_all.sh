#!/usr/bin/env bash
set -euo pipefail

# Run all frameworks serially so no benchmark can inherit GPU contention from
# another. This script runs on the benchmark host, not inside a container.
bench_root=${BENCH_ROOT:-/root/opd-bench}
artifact_root=${ARTIFACT_ROOT:-$bench_root/artifacts/mopd-frameworks}
steps=${STEPS:-15}
teacher_mode=${TEACHER_MODE:-multi}
run_id=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}

STEPS="$steps" RUN_ID="$run_id" TEACHER_MODE="$teacher_mode" BENCH_ROOT="$bench_root" \
  bash "$artifact_root/run_prime.sh"

docker exec -i \
  -e STEPS="$steps" -e RUN_ID="$run_id" -e TEACHER_MODE="$teacher_mode" \
  -e BENCH_ROOT="$bench_root" \
  opd-miles bash "$artifact_root/run_miles.sh"

STEPS="$steps" RUN_ID="$run_id" TEACHER_MODE="$teacher_mode" BENCH_ROOT="$bench_root" \
  bash "$artifact_root/run_verl.sh"

printf '%s\n' \
  "$bench_root/mopd/runs/prime-$run_id" \
  "$bench_root/mopd/runs/miles-$run_id" \
  "$bench_root/mopd/runs/verl-$run_id"
