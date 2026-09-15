"""Own one Slurm allocation and stop its two container steps after training."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
IMAGE = "@CONTAINER_IMAGE@"
PREVIOUS = "@ASSETS@"
BASE = "@BASE_PARENT@"


def stop(*_):
    raise SystemExit(143)


def main():
    job = os.environ["SLURM_JOB_ID"]
    result = ROOT / "results" / f"slime-opd-{job}"
    result.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "MOPD_RESULT": str(result), "MOPD_CAMPAIGN": str(ROOT),
           "NVIDIA_VISIBLE_DEVICES": "all", "NVIDIA_DRIVER_CAPABILITIES": "all"}
    nodes = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True).split()
    assert nodes == ["@GENERATION_NODE@", "@TRAINER_NODE@"], nodes
    (ROOT / "active-run.json").write_text(json.dumps({"job_id": job, "result": str(result), "nodes": nodes}))
    (result / "start.json").write_text(json.dumps({"time": time.time(), "job": job, "nodes": nodes}))
    (result / "slurm-start.txt").write_bytes(subprocess.check_output(["scontrol", "show", "job", job]))
    subprocess.run(["tar", "--exclude=__pycache__", "--exclude=.git", "--exclude=.venv", "-czf", str(result / "launch-source.tar.gz"), "-C", str(ROOT),
                    "source", "data", "control.py", "node.py", "observe.py", "campaign.py",
                    "serve-teachers.py", "teacher-provenance.json", "test_campaign.py", "make_args.py",
                    "endpoints.py", "scoring.py", "train_entry.py", "source-provenance.json", "launch.py", "mismatch-metrics.yaml", "wheels.sha256", "test_weight_session.py", "test_collectives.py"], check=True)
    common = ["srun", "--exclusive", "-N1", "-n1", "--gres=gpu:8", "--cpus-per-task=80",
              f"--container-image={IMAGE}", "--container-writable", "--container-remap-root",
              "--no-container-mount-home", f"--container-mounts={ROOT}:{ROOT},{PREVIOUS}:{PREVIOUS}:ro,"
              f"@MEGATRON_PARENT@:@MEGATRON_PARENT@:ro,{BASE}:{BASE}:ro"]
    children = []
    code = 1
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for role, node in (("learner", nodes[1]), ("generation", nodes[0])):
            command = common + ["-w", node, "python3", str(ROOT / "node.py"), role]
            (result / f"{role}-srun.json").write_text(json.dumps(command, indent=2))
            log = (result / f"{role}-driver.out").open("wb")
            children.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        while True:
            if children[0].poll() is not None:
                code = children[0].returncode
                break
            if children[1].poll() is not None:
                raise RuntimeError(f"Generation step exited early with {children[1].returncode}")
            time.sleep(5)
    finally:
        (result / "STOP").touch()
        deadline = time.monotonic() + 40
        while any(p.poll() is None for p in children) and time.monotonic() < deadline:
            time.sleep(1)
        for p in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in children:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        (result / "exit-code.txt").write_text(str(code) + "\n")
        (result / "end.json").write_text(json.dumps({"time": time.time(), "exit_code": code}))
    return code


if __name__ == "__main__":
    sys.exit(main())
