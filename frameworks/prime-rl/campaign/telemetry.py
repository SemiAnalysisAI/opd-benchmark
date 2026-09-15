"""Collect lightweight node and inference measurements without GPU profiling."""

import concurrent.futures
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

result, role = Path(sys.argv[1]), sys.argv[2]
stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stop.set())
signal.signal(signal.SIGINT, lambda *_: stop.set())


def prometheus(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=2) as response:
            return response.read().decode()
    except OSError as error:
        return {'error': str(error)}


with gzip.open(result / f'{role}-telemetry.jsonl.gz', 'at') as output:
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        while not stop.is_set() and not (result / 'STOP').exists():
            started = time.monotonic()
            row = {'time_unix': time.time(), 'role': role, 'host_scope': 'whole node'}
            command = ['nvidia-smi', '--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem', '--format=csv']
            try:
                row['gpu_csv'] = subprocess.check_output(command, text=True, timeout=4)
            except (subprocess.SubprocessError, OSError) as error:
                row['gpu_error'] = str(error)
            for name in ('stat', 'meminfo', 'net/dev', 'diskstats', 'loadavg'):
                try:
                    row[name] = Path('/proc', name).read_text()
                except OSError as error:
                    row[name] = str(error)
            if role == 'generation':
                ports = (8100, 8110, 8120, 8200, 8300, 29000)
                row['prometheus'] = dict(zip(map(str, ports), pool.map(prometheus, ports)))
            output.write(json.dumps(row) + '\n')
            output.flush()
            stop.wait(max(0.0, 5 - (time.monotonic() - started)))
