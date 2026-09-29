#!/bin/bash
# Sample InfiniBand port counters with perfquery, for pods whose sysfs has no ports/*/counters (the ClusterMAX
# fabric_monitor.sh then records empty IB rows). The CSV has fabric_monitor.sh's IB columns, and PortXmitData and
# PortRcvData are in 4-byte words like the sysfs counters, so ClusterMAX's summarize_ib reads it unchanged;
# summarize.py uses telemetry/fabric/ib-perfquery in place of telemetry/fabric/ib when it exists.
set -u
: "${MILES_FABRIC_TELEMETRY_DIR:?MILES_FABRIC_TELEMETRY_DIR must be set}"
: "${IB_SAMPLE_SECONDS:=5}"
node="$(hostname -s 2>/dev/null || hostname)"
out="$MILES_FABRIC_TELEMETRY_DIR/ib-perfquery/${node}.csv"
mkdir -p "$(dirname "$out")"
printf 'collector_time_utc,node,device,port,state,rate,port_xmit_data,port_rcv_data,port_xmit_packets,port_rcv_packets,symbol_error,link_downed,port_rcv_errors,port_xmit_discards\n' > "$out"
devices=()
for d in /sys/class/infiniband/*; do
    [[ $(cat "$d/ports/1/link_layer" 2>/dev/null) == InfiniBand ]] && devices+=("$(basename "$d")")
done
while true; do
    stamp="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
    for dev in "${devices[@]}"; do
        state=$(cat "/sys/class/infiniband/$dev/ports/1/state" 2>/dev/null | tr -d ' ')
        rate=$(cat "/sys/class/infiniband/$dev/ports/1/rate" 2>/dev/null | tr -d ' ')
        perfquery -x -C "$dev" -P 1 2>/dev/null | awk -v t="$stamp" -v n="$node" -v d="$dev" -v s="$state" -v r="$rate" -F'[:.]+' '
            /^PortXmitData/ {x=$NF} /^PortRcvData/ {rx=$NF} /^PortXmitPkts/ {xp=$NF} /^PortRcvPkts/ {rp=$NF}
            END { if (x != "") printf "%s,%s,%s,1,\"%s\",\"%s\",%s,%s,%s,%s,,,,\n", t, n, d, s, r, x, rx, xp, rp }' >> "$out"
    done
    sleep "$IB_SAMPLE_SECONDS"
done
