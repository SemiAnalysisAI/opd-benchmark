#!/usr/bin/env bash
set -euo pipefail

# Run locally after collect_environment.sh and removal of Prime broadcast weights.
remote=${REMOTE:-root@31.22.104.163}
bench_root=${BENCH_ROOT:-/dev/shm/opd-bench}
run_id=${RUN_ID:-b200-full}
destination=${DESTINATION:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results}
mkdir -p "$destination/all-runs" "$destination/environment"

compact_filters=(
  --include='*/'
  --include='*.log'
  --include='*.env'
  --include='*.csv'
  --include='*.json'
  --include='*.jsonl'
  --include='*.toml'
  --include='*.yaml'
  --include='*.patch'
  --include='*.txt'
  --exclude='*'
)

rsync -a --prune-empty-dirs "${compact_filters[@]}" \
  "$remote:$bench_root/runs-full/" "$destination/all-runs/"
rsync -a --prune-empty-dirs "${compact_filters[@]}" \
  "$remote:$bench_root/logs/" "$destination/setup-logs/"
rsync -a --prune-empty-dirs "${compact_filters[@]}" \
  "$remote:$bench_root/results-full/environment/" "$destination/environment/"
scp -q "$remote:$bench_root/full-driver.log" "$destination/full-driver.log"

for framework in prime miles verl; do
  source_dir="$destination/all-runs/$framework-$run_id"
  target_dir="$destination/$framework"
  test -d "$source_dir"
  rm -rf -- "$target_dir"
  mkdir -p "$target_dir"
  find "$source_dir" -maxdepth 1 -type f -exec cp {} "$target_dir/" \;
done

# The analyzer uses stable names for Prime's two metric streams.
cp "$destination/all-runs/prime-$run_id/output/metrics.jsonl" \
  "$destination/prime/trainer-metrics.jsonl"
cp "$destination/all-runs/prime-$run_id/output/run_default/metrics.jsonl" \
  "$destination/prime/orchestrator-metrics.jsonl"
cp "$destination/all-runs/prime-$run_id/output/run_default/final_summary.json" \
  "$destination/prime/final_summary.json"

if find "$destination" -type f \( \
  -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' -o \
  -name '*.ckpt' \) -print -quit | grep -q .; then
  printf 'Model artifact unexpectedly entered compact result set\n' >&2
  exit 1
fi
if find "$destination" -type f -size +100M -print -quit | grep -q .; then
  printf 'A compact result file exceeds 100 MB\n' >&2
  exit 1
fi

printf '%s\n' "$destination"
