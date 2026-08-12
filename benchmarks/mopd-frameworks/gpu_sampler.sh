#!/usr/bin/env bash
set -euo pipefail

output=${1:?usage: gpu_sampler.sh OUTPUT.csv}
printf '%s\n' 'epoch,index,uuid,name,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_w' > "$output"
while true; do
  epoch=$(date +%s)
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits | awk -v epoch="$epoch" -F ', ' \
    '{print epoch "," $1 "," $2 "," $3 "," $4 "," $5 "," $6 "," $7}' >> "$output"
  sleep 1
done
