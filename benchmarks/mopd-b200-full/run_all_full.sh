#!/usr/bin/env bash
set -euo pipefail

bench_root=${BENCH_ROOT:-/dev/shm/opd-bench}
artifact_root=${ARTIFACT_ROOT:-$bench_root/artifacts/mopd-b200-full}
steps=${STEPS:-15}
run_id=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}

STEPS="$steps" RUN_ID="$run_id" BENCH_ROOT="$bench_root" ARTIFACT_ROOT="$artifact_root" \
  bash "$artifact_root/run_prime_full.sh"
docker restart opd-miles-full >/dev/null
docker exec -i -e STEPS="$steps" -e RUN_ID="$run_id" -e BENCH_ROOT="$bench_root" \
  -e ARTIFACT_ROOT="$artifact_root" opd-miles-full bash "$artifact_root/run_miles_full.sh"
STEPS="$steps" RUN_ID="$run_id" BENCH_ROOT="$bench_root" ARTIFACT_ROOT="$artifact_root" \
  bash "$artifact_root/run_verl_full.sh"

printf '%s\n' \
  "$bench_root/runs-full/prime-$run_id" \
  "$bench_root/runs-full/miles-$run_id" \
  "$bench_root/runs-full/verl-$run_id"
