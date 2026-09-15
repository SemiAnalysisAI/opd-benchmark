"""Scrape only the generation endpoints advertised by this campaign's launch log."""

import gzip
import json
from pathlib import Path
import re
import signal
import sys
import threading
import time
import urllib.request

root = Path(__file__).resolve().parent
job = sys.argv[1]
assert job.isdecimal()
result = root / "results" / f"mopd-async-{job}"
stopping = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stopping.set())
signal.signal(signal.SIGINT, lambda *_: stopping.set())
ports = set()
offset = 0
with gzip.open(result / "generation-endpoints.jsonl.gz", "at") as output:
    while not stopping.is_set() and not (result / "end.json").exists():
        if (result / "learner-train.out").exists():
            with (result / "learner-train.out").open() as log:
                log.seek(offset)
                for line in log:
                    if "sglang.launch_server" in line and str(root / "models") in line:
                        ports.update(int(value) for value in re.findall(r"--port[ =]+(\d+)", line))
                offset = log.tell()
        if (result / "generation-ready.json").exists():
            ip = json.loads((result / "generation-ready.json").read_text())["ip"]
            assert ip == "@GENERATION_IP@"
            for port in sorted(ports):
                assert 1024 < port < 65536
                row = {"time_unix": time.time(), "ip": ip, "port": port}
                try:
                    with urllib.request.urlopen(f"http://{ip}:{port}/metrics", timeout=2) as response:
                        row["prometheus"] = response.read().decode()
                except OSError as error:
                    row["error"] = str(error)
                output.write(json.dumps(row) + "\n")
            output.flush()
        (result / "generation-endpoints-status.json").write_text(json.dumps({"time_unix": time.time(), "ports": sorted(ports), "finished": False}))
        if (result / "STOP").exists():
            break
        stopping.wait(10)
(result / "generation-endpoints-status.json").write_text(json.dumps({"time_unix": time.time(), "ports": sorted(ports), "finished": True}))
