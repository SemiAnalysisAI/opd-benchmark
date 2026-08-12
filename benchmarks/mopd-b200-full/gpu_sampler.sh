#!/usr/bin/env bash
set -euo pipefail

output=${1:?usage: gpu_sampler.sh OUTPUT.csv}
printf '%s\n' 'epoch,index,utilization_gpu_pct,memory_used_mib,power_w' > "$output"
while true; do
  epoch=$(date +%s)
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader,nounits | while IFS=, read -r index util memory power; do
      printf '%s,%s,%s,%s,%s\n' "$epoch" "${index// /}" "${util// /}" \
        "${memory// /}" "${power// /}" >> "$output"
    done
  sleep 1
done
