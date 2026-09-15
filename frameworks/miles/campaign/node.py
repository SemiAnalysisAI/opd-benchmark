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
    tests = ["tests/fast/rollout/test_fully_async_rollout.py",
             "tests/fast/backends/training_utils/loss/test_candidate_opd.py",
             "tests/fast/examples/test_mopd_data_source.py",
             "tests/fast/examples/test_mopd_puzzles.py", str(ROOT / "test-balanced-integration.py")]
    with (RESULT / "preflight-tests.out").open("wb") as log:
        subprocess.run([sys.executable, str(ROOT / "test-async-contract.py")], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, str(ROOT / "test-scoring-retry.py")], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, str(ROOT / "test-balanced-buffer.py")], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        subprocess.run([sys.executable, "-c", "from types import SimpleNamespace; from miles.utils.tracking_utils.tensorboard_utils import _TensorboardAdapter; t=_TensorboardAdapter(SimpleNamespace(use_tensorboard=True,tb_project_name='async',tb_experiment_name='preflight')); t.log({'smoke':1.0},0); t.finish()"],
                       env={**os.environ, "TENSORBOARD_DIR": str(RESULT / "preflight-tensorboard")},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    (RESULT / "preflight-passed.json").write_text(json.dumps({"time": time.time(), "tests": tests}))


def verify_placement(head_ip, generation_ip):
    import ray
    from ray.util import remove_placement_group
    from miles.ray.placement_group import _create_placement_group

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
        actual = [int(x) for x in group.pg_reordered_gpu_ids]
        assert actual == list(range(8)) + list(range(2, 8)), actual
        (RESULT / "placement-verified.json").write_text(json.dumps({"nodes": nodes, "sorted_physical_gpu_ids": actual}))
    finally:
        remove_placement_group(group.pg)
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
    extra = ["--sglang-moe-runner-backend", "triton", "--sglang-enable-metrics",
             "--custom-async-data-buffer-path", "balanced_async_buffer.BalancedAsyncBuffer",
             "--custom-rm-path", "campaign_async.reward_func", "--eval-config", str(ROOT / "eval.json"),
             "--async-max-concurrent-samples", "256", "--async-data-buffer-capacity-factor", "1",
             "--max-weight-staleness", "2", "--update-weights-interval", "1",
             "--save-debug-event-data", str(RESULT / "events"),
             "--use-tensorboard", "--tb-project-name", str(RESULT / "tensorboard"),
             "--tb-experiment-name", "async-mopd", "--no-offload-train", "--no-offload-rollout"]
    command = [sys.executable, "scripts/run_mopd_puzzles.py", "--mode", "student", "--num-nodes", "2",
               "--fully-async", "--no-colocate", "--use-rollout-logprobs", "--actor-gpus", "8",
               "--megatron-path", "@MEGATRON_SOURCE@",
               "--rollout-gpus", "6", "--rollout-gpus-per-engine", "2", "--model-dir", str(ROOT / "models"),
               "--data-dir", str(ROOT / "data"), "--checkpoint-dir", str(ROOT / "checkpoints" / RESULT.name),
               "--teacher-urls", f"countdown=http://{ready['ip']}:30000/generate graph_color=http://{ready['ip']}:30001/generate",
               "--candidate-top-k", "16", "--loss-mode", "topk-candidate", "--reward-refresh",
               "--domain-balance", "static", "--num-rollout", "40", "--rollout-batch-size", "128",
               "--n-samples-per-prompt", "1", "--global-batch-size", "128", "--eval-interval", "10",
               "--save-interval", "20", "--no-cleanup-processes", "--no-sparse-scoring",
               "--extra-env-vars", json.dumps({"MOPD_RESULT": str(RESULT), "MOPD_CAMPAIGN": str(ROOT),
                                               "TENSORBOARD_DIR": str(RESULT / "tensorboard"),
                                               "PYTHONPATH": os.environ["PYTHONPATH"], "NCCL_DEBUG": "WARN"}),
               "--extra-args", shlex.join(extra)]
    training = launch(command, "train")
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
    sys.path.insert(0, str(ROOT / "source"))
    ip = '@TRAINER_IP@' if ROLE == 'learner' else '@GENERATION_IP@'
    ray_tmp = Path(f"/tmp/mopd-async-{os.environ['SLURM_JOB_ID']}-{ROLE}")
    os.environ.update({"PYTHONPATH": f"{ROOT}:{ROOT}/source:@MEGATRON_SOURCE@", "PYTHONNOUSERSITE": "1",
                       "PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled", "NCCL_DEBUG": "WARN",
                       "MASTER_ADDR": ip, "RAY_TMPDIR": str(ray_tmp), "MILES_SCRIPT_EXTERNAL_RAY": "1",
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
