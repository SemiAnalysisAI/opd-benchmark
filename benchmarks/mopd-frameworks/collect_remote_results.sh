#!/usr/bin/env bash
set -euo pipefail

# Download the compact result set used by analyze_results.py. Run locally.
remote=${REMOTE:-root@86.38.238.166}
bench_root=${BENCH_ROOT:-/root/opd-bench}
run_id=${RUN_ID:-full}
teacher_mode=${TEACHER_MODE:-multi}
destination=${DESTINATION:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results}

case "$teacher_mode" in
  single|multi) ;;
  *) printf 'TEACHER_MODE must be single or multi, got %s\n' "$teacher_mode" >&2; exit 2 ;;
esac

mkdir -p "$destination/prime" "$destination/miles" "$destination/verl"

copy_file() {
  local framework=$1 source_name=$2 destination_name=${3:-$2}
  scp -q "$remote:$bench_root/mopd/runs/$framework-$run_id/$source_name" \
    "$destination/$framework/$destination_name"
}

for name in run.log wall_time.env gpu.csv math-teacher.log; do
  copy_file prime "$name"
  copy_file miles "$name"
  copy_file verl "$name"
done
if [[ "$teacher_mode" == multi ]]; then
  for framework in prime miles verl; do
    copy_file "$framework" code-teacher.log
  done
fi
copy_file miles ray.log

if ssh -q "$remote" test -f "$bench_root/mopd/runs/slime-$run_id/wall_time.env"; then
  mkdir -p "$destination/slime"
  for name in run.log wall_time.env gpu.csv math-teacher.log ray.log layout.env; do
    copy_file slime "$name"
  done
  if [[ "$teacher_mode" == multi ]]; then
    copy_file slime code-teacher.log
  fi
fi

scp -q "$remote:$bench_root/mopd/runs/prime-$run_id/output/metrics.jsonl" \
  "$destination/prime/trainer-metrics.jsonl"
scp -q "$remote:$bench_root/mopd/runs/prime-$run_id/output/run_default/metrics.jsonl" \
  "$destination/prime/orchestrator-metrics.jsonl"
scp -q "$remote:$bench_root/mopd/runs/prime-$run_id/output/run_default/final_summary.json" \
  "$destination/prime/final_summary.json"

printf '%s\n' "$destination"
