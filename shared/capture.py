"""Sample GPU, host and Prometheus metrics until the campaign stops.

    python -m shared.capture ROLE [HOST:PORT ...] [--engine-log LOG --engine-host HOST ...] [--vllm]

Every five seconds, appends one row to `<result>/<ROLE>-capture.jsonl.gz`.
Host counters cover the whole node. `--engine-log` follows a Miles or Slime
training log and adds each SGLang policy engine it announces on an `--engine-host` as a target.
`--vllm` adds the HTTP server of each vLLM engine that Ray starts on this node (verl's policy
and teacher servers): it probes only TCP ports that a `vLLMHttpServer` actor process listens on.
"""
import argparse
import concurrent.futures
import gzip
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import urllib.request

from shared.slurm import STOP_FILE, result_dir

INTERVAL_SECONDS = 5
GPU_QUERY = ['nvidia-smi', '--format=csv', '--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,'
             'memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem']
PROC_FILES = ('stat', 'meminfo', 'net/dev', 'diskstats', 'loadavg')


def engine_ports(line, host):
    """Ports of the policy engines on `host` announced in one Miles or Slime log line."""
    ports = set()
    if 'sglang.launch_server' in line and '--model-path' in line:
        ports.update(int(v) for v in re.findall(r'--port[ =]+(\d+)', line))
    ports.update(int(v) for v in re.findall(rf'Launch HttpServerEngineAdapter at: {re.escape(host)}:(\d+)', line))
    if 'Ports for engine' in line and f"'host': '{host}'" in line:
        ports.update(int(v) for v in re.findall(r"'port': (\d+)", line))
    return ports


VLLM_PROCESS = b'vLLMHttpServer'  # Ray titles each actor process after its class.
VLLM_PROBES = 3


def listening_sockets():
    """{socket inode: 'IP:PORT'} for every IPv4 TCP socket listening on this node."""
    sockets = {}
    for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
        fields = line.split()
        if fields[3] != '0A':  # TCP_LISTEN
            continue
        address, port = fields[1].split(':')
        ip = '.'.join(str(int(address[i:i + 2], 16)) for i in (6, 4, 2, 0))
        sockets[fields[9]] = f'{"127.0.0.1" if ip == "0.0.0.0" else ip}:{int(port, 16)}'
    return sockets


def vllm_candidates():
    """'IP:PORT' of the listening sockets owned by vLLM server actor processes."""
    listening, found = listening_sockets(), set()
    for pid in filter(str.isdigit, os.listdir('/proc')):
        try:
            if VLLM_PROCESS not in Path(f'/proc/{pid}/cmdline').read_bytes():
                continue
            for fd in Path(f'/proc/{pid}/fd').iterdir():
                link = os.readlink(fd)
                if link.startswith('socket:[') and link[8:-1] in listening:
                    found.add(listening[link[8:-1]])
        except OSError:  # The process exited, or its file descriptors are not readable.
            continue
    return found


def read(function, *args, **kwargs):
    """Call `function`, recording a failure in the row instead of stopping the capture."""
    try:
        return function(*args, **kwargs)
    except (OSError, subprocess.SubprocessError) as error:
        return {'error': str(error)}


def prometheus(target):
    with urllib.request.urlopen(f'http://{target}/metrics', timeout=2) as response:
        return response.read().decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('role', help='names the output file, e.g. learner or generation')
    parser.add_argument('targets', nargs='*', help='HOST:PORT Prometheus endpoints')
    parser.add_argument('--engine-log', type=Path, help='training log to scan for policy-engine ports')
    parser.add_argument('--engine-host', nargs='+', default=[], help='hosts whose engines to scrape from --engine-log')
    parser.add_argument('--vllm', action='store_true', help="scrape the vLLM servers Ray starts on this node")
    args = parser.parse_args()
    result = result_dir()
    targets, offset = set(args.targets), 0
    probed, next_probe = {}, 0.0  # --vllm: tries per candidate, and when to look for new ones
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    with gzip.open(result / f'{args.role}-capture.jsonl.gz', 'at') as output, \
            concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        while not stop.is_set() and not (result / STOP_FILE).exists():
            started = time.monotonic()
            if args.engine_log and args.engine_log.exists():
                with args.engine_log.open() as log:
                    log.seek(offset)
                    for line in log:
                        targets |= {f'{host}:{p}' for host in args.engine_host for p in engine_ports(line, host)}
                    offset = log.tell()
            if args.vllm and started >= next_probe:
                # A server's other listening sockets (e.g. its engine's RPC ports) do not serve vLLM metrics.
                # A candidate is retried a few times in case a busy server misses the timeout.
                candidates = read(vllm_candidates)
                for candidate in candidates if isinstance(candidates, set) else ():
                    if candidate not in targets and probed.get(candidate, 0) < VLLM_PROBES:
                        probed[candidate] = probed.get(candidate, 0) + 1
                        body = read(prometheus, candidate)
                        if isinstance(body, str) and 'vllm:' in body:
                            targets.add(candidate)
                next_probe = started + 30
            ordered = sorted(targets)
            row = {'time_unix': time.time(), 'role': args.role, 'host_scope': 'whole node',
                   'gpu_csv': read(subprocess.check_output, GPU_QUERY, text=True, timeout=4),
                   **{name: read(Path('/proc', name).read_text) for name in PROC_FILES},
                   'prometheus': dict(zip(ordered, pool.map(lambda t: read(prometheus, t), ordered)))}
            output.write(json.dumps(row) + '\n')
            output.flush()
            stop.wait(max(0.0, INTERVAL_SECONDS - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
