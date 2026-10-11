# Cluster steps that every self-hosted server launcher shares; sourced by <framework>/serve.sbatch inside a Slurm job.
#
# selfhosted_setup FRAMEWORK IMAGE CONTAINER_NAME
#     Sets RUN ($CAMPAIGN/<framework>/server-<job>), NODES, HEAD, HEAD_IP and CONTAINER (the pyxis flags); removes
#     leaked /dev/shm segments; writes node-ips.json; starts the GPU, fabric and InfiniBand samplers and the
#     before-snapshot on every node; exports the NCCL settings.
# selfhosted_ray PYTHON_BIN_DIR
#     Starts a Ray head on HEAD and a worker on every other node, inside the container, and waits until every GPU
#     has joined.
# selfhosted_finish
#     Takes the after-snapshot and removes leaked /dev/shm segments.
: "${CAMPAIGN:?CAMPAIGN must be set}"
SELFHOSTED=$CAMPAIGN/opd/hosted/selfhosted

selfhosted_setup() {
    local framework=$1 image=$2 name=$3
    RUN=$CAMPAIGN/$framework/server-$SLURM_JOB_ID
    mkdir -p "$RUN/logs"
    mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    HEAD=${NODES[0]}
    HEAD_IP=$(srun --overlap -N1 -w "$HEAD" hostname -I | awk '{print $1}')
    # node -> IP, so summarize.py can map engine and trainer addresses in server logs to nodes.
    srun --overlap -N"${#NODES[@]}" --ntasks-per-node=1 bash -c 'echo "$(hostname -s) $(hostname -I | cut -d" " -f1)"' \
        | python3 -c 'import json, sys; print(json.dumps(dict(l.split() for l in sys.stdin if l.strip()), indent=2))' \
        > "$RUN/node-ips.json"
    # The pods' /dev/shm is 16 GiB, and killed vLLM, SGLang and PyTorch workers leave their segments in it; NCCL
    # then fails to create its own. Remove every segment no process holds.
    srun --overlap -N"${#NODES[@]}" --ntasks-per-node=1 python3 "$SELFHOSTED/shm_clean.py"

    export MILES_GPU_TELEMETRY_DIR=$RUN/telemetry/gpu MILES_FABRIC_TELEMETRY_DIR=$RUN/telemetry/fabric
    export MILES_SYSTEM_TELEMETRY_DIR=$RUN/telemetry/system
    srun --overlap -N"${#NODES[@]}" --ntasks-per-node=1 bash "$CAMPAIGN/telemetry-scripts/system_snapshot.sh" before
    srun --overlap -N"${#NODES[@]}" --ntasks-per-node=1 bash "$CAMPAIGN/telemetry-scripts/gpu_monitor.sh" &
    srun --overlap -N"${#NODES[@]}" --ntasks-per-node=1 bash "$CAMPAIGN/telemetry-scripts/fabric_monitor.sh" &
    # The pods' sysfs has no InfiniBand port counters, so fabric_monitor.sh records empty IB rows; perfquery reads them.
    srun --overlap -N"${#NODES[@]}" --ntasks-per-node=1 bash "$SELFHOSTED/ib_monitor.sh" &

    export NVIDIA_VISIBLE_DEVICES=all HF_HOME=$CAMPAIGN/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_DEBUG_FILE=$RUN/logs/nccl-%h-%p.out
    # Images built for AWS load the aws-ofi-nccl plugin, which finds no EFA device and falls back to Libfabric TCP
    # over the pod network. Use NCCL's own InfiniBand transport on the eight 400 Gb/s NDR rails (NCCL_NET=IB fails
    # instead of falling back); override NCCL_IB_HCA on other clusters.
    export NCCL_NET_PLUGIN=none NCCL_NET=IB
    export NCCL_IB_HCA=${NCCL_IB_HCA:-=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_9,mlx5_12,mlx5_13}
    CONTAINER=(--container-image="$image" --container-name="$name" --container-mounts=/shared:/shared
               --container-writable --export=ALL)
}

selfhosted_ray() {
    local bin=$1 node gpus want=$((${#NODES[@]} * 8))
    srun --overlap -N1 -w "$HEAD" "${CONTAINER[@]}" bash -c \
        "$bin/ray start --head --node-ip-address=$HEAD_IP --port=6379 --disable-usage-stats --block" \
        > "$RUN/logs/ray-$HEAD.log" 2>&1 &
    sleep 20
    for node in "${NODES[@]:1}"; do
        srun --overlap -N1 -w "$node" "${CONTAINER[@]}" bash -c \
            "$bin/ray start --address=$HEAD_IP:6379 --node-ip-address=\$(hostname -I | awk '{print \$1}') --disable-usage-stats --block" \
            > "$RUN/logs/ray-$node.log" 2>&1 &
    done
    # Serve only once every GPU has joined; a node whose container failed to start would otherwise leave a
    # placement group pending until the first request times out.
    for _ in $(seq 60); do
        gpus=$(srun --overlap -N1 -w "$HEAD" "${CONTAINER[@]}" "$bin/python3" -c \
            "import ray; ray.init(address='$HEAD_IP:6379', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null | tail -1)
        [[ $gpus == "$want" ]] && return 0
        sleep 10
    done
    echo "Ray has ${gpus:-0} of $want GPUs; see $RUN/logs/ray-*.log" >&2
    exit 1
}

selfhosted_finish() {
    srun --overlap -N"${#NODES[@]}" --ntasks-per-node=1 bash "$CAMPAIGN/telemetry-scripts/system_snapshot.sh" after
    srun --overlap -N"${#NODES[@]}" --ntasks-per-node=1 python3 "$SELFHOSTED/shm_clean.py"
}
