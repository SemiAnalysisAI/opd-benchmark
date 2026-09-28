"""Sample GPU, host and Prometheus metrics until the campaign stops.

    python -m shared.capture ROLE [HOST:PORT ...] [--engine-log LOG --engine-host HOST]

Every five seconds, appends one row to `<result>/<ROLE>-capture.jsonl.gz`.
Host counters cover the whole node. `--engine-log` follows a Miles or Slime
training log and adds each SGLang policy engine it announces as a target.
"""
import argparse
import concurrent.futures
import gzip
import json
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
    parser.add_argument('--engine-host', help='host whose engines to scrape from --engine-log')
    args = parser.parse_args()
    result = result_dir()
    targets, offset = set(args.targets), 0
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
                        targets |= {f'{args.engine_host}:{p}' for p in engine_ports(line, args.engine_host)}
                    offset = log.tell()
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
