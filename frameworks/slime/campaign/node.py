"""Run one owned container role with lightweight measurement and bounded cleanup."""

import ipaddress
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(os.environ["MOPD_CAMPAIGN"])
RESULT = Path(os.environ["MOPD_RESULT"])
ROLE = sys.argv[1]
CHILDREN = []


def launch(command, name, extra=None):
    (RESULT / f"{ROLE}-{name}-command.json").write_text(json.dumps(command, indent=2))
    process = subprocess.Popen(command, env={**os.environ, **(extra or {})},
                               stdout=(RESULT / f"{ROLE}-{name}.out").open("wb"),
                               stderr=subprocess.STDOUT, start_new_session=True)
    CHILDREN.append(process)
    return process


def check_children(ignore=None):
    if (RESULT / "STOP").exists():
        raise SystemExit(143)
    if any(p is not ignore and p.poll() is not None for p in CHILDREN):
        raise RuntimeError("A required process exited; inspect the per-process logs")


def wait_file(name, timeout=900):
    deadline = time.monotonic() + timeout
    while not (RESULT / name).exists():
        check_children()
        if time.monotonic() > deadline:
            raise TimeoutError(f"Readiness timed out for {name}")
        time.sleep(2)
    return json.loads((RESULT / name).read_text())


def stop(*_):
    raise SystemExit(143)


def preflight():
    tests = [str(ROOT / 'test_campaign.py'), str(ROOT / 'test_weight_session.py'), str(ROOT / 'train_entry.py')]
    with (RESULT / "preflight-tests.out").open("wb") as log:
        for test in tests:
            subprocess.run([sys.executable, test], env={**os.environ, 'MOPD_PARSE_ONLY':'1'},
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
                        str(ROOT/'test_collectives.py')], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=150)
    (RESULT / "preflight-passed.json").write_text(json.dumps({"time": time.time(), "tests": tests}))


def verify_placement(head_ip, generation_ip):
    import ray
    from ray.util import remove_placement_group
    from slime.ray.placement_group import _create_placement_group

    ray.init(address=f"{head_ip}:6379")
    deadline = time.monotonic() + 180
    while int(ray.cluster_resources().get("GPU", 0)) != 14:
        check_children()
        assert time.monotonic() < deadline, ray.cluster_resources()
        time.sleep(2)
    nodes = {n["NodeManagerAddress"]: n["Resources"].get("GPU") for n in ray.nodes() if n["Alive"]}
    assert nodes == {head_ip: 8.0, generation_ip: 6.0}, nodes
    assert ipaddress.ip_address(head_ip) < ipaddress.ip_address(generation_ip), nodes
    group = _create_placement_group(14)
    try:
        actual = [int(x) for x in group[2]]
        assert actual == list(range(8)) + list(range(2, 8)), actual
        (RESULT / "placement-verified.json").write_text(json.dumps({"nodes": nodes, "sorted_physical_gpu_ids": actual}))
    finally:
        remove_placement_group(group[0])
        ray.shutdown()


def learner(head_ip):
    preflight()
    launch(["ray", "start", "--head", f"--node-ip-address={head_ip}", "--num-gpus=8", "--num-cpus=64",
            "--disable-usage-stats", "--block"], "ray")
    (RESULT / "head-started.json").write_text(json.dumps({"ip": head_ip}))
    ready = wait_file("generation-ready.json")
    verify_placement(head_ip, ready["ip"])
    (RESULT / "placement-approved.json").write_text(json.dumps({"time": time.time()}))
    wait_file("teachers-ready.json")
    urls = {domain: f"http://{ready['ip']}:{port}/generate"
            for domain, port in [('countdown',30000), ('graph_color',30001)]}
    training = launch([sys.executable, str(ROOT / 'train_entry.py')], 'train',
                      {'RAY_ADDRESS': f'{head_ip}:6379', 'MOPD_TEACHER_URLS': json.dumps(urls)})
    while training.poll() is None:
        check_children(ignore=training)
        time.sleep(3)
    return training.returncode


def generation(ip):
    wait_file("preflight-passed.json")
    head = wait_file("head-started.json")
    # The head writes its address before its GCS socket becomes ready.
    deadline = time.monotonic() + 180
    while True:
        try:
            with socket.create_connection((head["ip"], 6379), timeout=2):
                break
        except OSError:
            check_children()
            assert time.monotonic() < deadline
            time.sleep(2)
    launch(["ray", "start", f"--address={head['ip']}:6379", f"--node-ip-address={ip}",
            "--num-gpus=6", "--num-cpus=64", "--disable-usage-stats", "--block"], "ray",
           {"CUDA_VISIBLE_DEVICES": "2,3,4,5,6,7"})
    (RESULT / "generation-ready.json").write_text(json.dumps({"ip": ip}))
    wait_file("placement-approved.json")
    launch([sys.executable, str(ROOT / "serve-teachers.py")], "teachers")
    while not (RESULT / "STOP").exists():
        check_children()
        time.sleep(3)
    return 0


def main():
    os.chdir(ROOT / "source")
    # Install the pinned wheel in this ephemeral container, without changing the shared image.
    with (RESULT / f'{ROLE}-numpy-install.out').open('wb') as log:
        subprocess.run(['sha256sum', '-c', str(ROOT/'wheels.sha256')], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall',
                        '--no-index', '--find-links', str(ROOT/'wheels'), 'numpy==1.26.4', 'scipy==1.15.3'],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    sys.path.insert(0, str(ROOT / "source"))
    ip = '@TRAINER_IP@' if ROLE == 'learner' else '@GENERATION_IP@'
    ray_tmp = Path(f"/tmp/mopd-async-{os.environ['SLURM_JOB_ID']}-{ROLE}")
    os.environ.update({"PYTHONPATH": f"{ROOT}:{ROOT}/source:@MEGATRON_SOURCE@", "PYTHONNOUSERSITE": "1",
                       "PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled", "NCCL_DEBUG": "WARN",
                       "MASTER_ADDR": ip, "RAY_TMPDIR": str(ray_tmp),
                       "TENSORBOARD_DIR": str(RESULT / 'tensorboard'), "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                       "SLIME_NATIVE_PROCESS_GROUPS": "1",
                       "RAY_memory_monitor_refresh_ms": "0", "RAY_DEDUP_LOGS": "0",
                       "TORCHINDUCTOR_CACHE_DIR": f"/tmp/mopd-inductor-{os.environ['SLURM_JOB_ID']}-{ROLE}"})
    os.environ.pop("NETRC", None)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    assert all(int(v) < 1024 for v in memory.split()), memory
    (RESULT / f"{ROLE}-gpu-before.txt").write_bytes(subprocess.check_output(["nvidia-smi", "-q"]))
    launch(["nvidia-smi", "--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem", "--format=csv", "-l", "5"], "gpu")
    launch([sys.executable, str(ROOT / "observe.py"), os.environ["SLURM_JOB_ID"], ROLE, str(os.getpid())], "observer")
    if ROLE == "learner":
        launch([sys.executable, str(ROOT / "endpoints.py"), os.environ["SLURM_JOB_ID"]], "endpoints")
    try:
        return learner(ip) if ROLE == "learner" else generation(ip)
    finally:
        for p in reversed(CHILDREN):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in CHILDREN:
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        copied = []
        for session in Path("/tmp").glob(f"mopd-async-{os.environ['SLURM_JOB_ID']}-*/ray/session_*"):
            if session.is_symlink() or not (session / "logs").exists():
                continue
            shutil.copytree(session / "logs", RESULT / "ray-logs" / ROLE, dirs_exist_ok=True)
            copied.append(str(session))
        (RESULT / f"{ROLE}-cleanup.json").write_text(json.dumps({"time": time.time(), "ray_sessions": copied}))


if __name__ == "__main__":
    sys.exit(main())
