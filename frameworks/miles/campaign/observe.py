"""Collect host counters and copy logs from one identified campaign container."""

import gzip
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import urllib.request

root = Path(__file__).resolve().parent
job, role, pid = sys.argv[1:]
result = root / "results" / f"mopd-async-{job}"
process = Path("/proc") / pid
assert (process / "cmdline").read_bytes().split(b"\0")[1:3] == [str(root / "node.py").encode(), role.encode()]
container_root = process / "root"
status = result / f"{role}-observer-status.json"
start = time.time()
last_copy = 0
iterations = 0
errors = []
stopping = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stopping.set())
signal.signal(signal.SIGINT, lambda *_: stopping.set())
with gzip.open(result / f"{role}-host-counters.jsonl.gz", "at") as stream:
    while process.exists() and not stopping.is_set() and not (result / "end.json").exists():
        now = time.time()
        row = {"time_unix": now, "scope": "node procfs counters, not per-job attribution"}
        for name in ("stat", "meminfo", "net/dev", "loadavg", "diskstats"):
            try:
                row[name] = (container_root / "proc" / name).read_text()
            except OSError as error:
                row[name] = str(error)
        if role == "generation":
            for port in (30000, 30001):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1) as response:
                        row[f"sglang_{port}"] = response.read().decode()
                except OSError as error:
                    row[f"sglang_{port}"] = str(error)
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        iterations += 1
        if now - last_copy >= 60 or (result / "STOP").exists():
            # Real session directories avoid absolute session_latest symlinks across mount namespaces.
            for logs in (container_root / "tmp").glob(f"mopd-async-{job}-*/ray/session_*/logs"):
                if logs.parent.is_symlink():
                    continue
                try:
                    shutil.copytree(logs, result / "ray-logs-live" / role, dirs_exist_ok=True,
                                    ignore=lambda directory, names: [name for name in names if Path(directory, name).is_symlink()])
                except (OSError, shutil.Error) as error:
                    errors.append(str(error)[:1000])
            last_copy = now
        status.write_text(json.dumps({"start": start, "last_sample": now, "iterations": iterations,
                                      "errors": errors[-5:], "finished": False}))
        if (result / "STOP").exists():
            break
        stopping.wait(10)
status.write_text(json.dumps({"start": start, "end": time.time(), "iterations": iterations,
                              "errors": errors[-5:], "finished": True}))
