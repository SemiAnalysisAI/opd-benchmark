#!/usr/bin/env bash
set -euo pipefail

output_path=${1:?usage: gpu_sampler.sh OUTPUT.csv}
interval=${GPU_SAMPLE_INTERVAL:-1}

printf '%s\n' 'timestamp,index,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_w,temperature_c' > "$output_path"
while true; do
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv,noheader,nounits >> "$output_path"
  sleep "$interval"
done

