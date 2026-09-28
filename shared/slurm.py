"""Paths, site values and process supervision for one prepared campaign.

`tools/prepare.py` copies this package to `<campaign>/shared/` and writes
`<campaign>/site.json`. Processes coordinate through files in the result
directory: `*-ready.json` files announce readiness and `STOP` asks all to exit.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from shared import recipe

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
PYDEPS = ROOT / 'pydeps'  # reasoning-gym for the scorer; see shared/scoring.py.
STOP_FILE = 'STOP'
STOP_EXIT_CODE = 143  # 128 + SIGTERM, as for a scheduler cancellation.
# Top-level campaign entries that are outputs, weight links or caches, not launch source.
# Compiled kernels (Triton, Inductor, FlashInfer, vLLM, SGLang) persist here, shared by both nodes and
# every run of the campaign, so GPUs do not wait on a cold compile after the first run.
CACHE = ROOT / 'runtime-cache'
CACHE_ENV = {'TRITON_CACHE_DIR': str(CACHE / 'triton'), 'TORCHINDUCTOR_CACHE_DIR': str(CACHE / 'inductor'),
             'FLASHINFER_WORKSPACE_BASE': str(CACHE), 'VLLM_CACHE_ROOT': str(CACHE / 'vllm'),
             'SGLANG_CACHE_DIR': str(CACHE / 'sglang')}
NOT_ARCHIVED = {'results', 'checkpoints', 'models', 'teachers', 'wheels', 'uv-cache', 'runtime-cache',
                'allocation.out', 'submission.lock', 'active-run.json'}


def site():
    return json.loads((ROOT / 'site.json').read_text())


def result_dir():
    return Path(os.environ['CAMPAIGN_RESULT'])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def exit_on_signals(code=STOP_EXIT_CODE):
    """Turn SIGTERM and SIGINT into SystemExit so that `finally` cleanup runs."""
    def stop(*_):
        raise SystemExit(code)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def eval_config(metadata_overrides=None):
    """Development evaluation in the Slime/Miles `--eval-config` format.

    Datasets inherit the training chat-template kwargs, so evaluation also runs with thinking on.
    """
    datasets = [{'name': d, 'path': str(DATA / recipe.data_file(d, 'dev')),
                 **({'metadata_overrides': metadata_overrides} if metadata_overrides else {})}
                for d in recipe.DOMAINS]
    return {'eval': {'defaults': {'temperature': 0, 'n_samples_per_eval_prompt': 1,
                                  'max_response_len': recipe.MAX_RESPONSE_TOKENS},
                     'datasets': datasets}}


class Supervisor:
    """Owned children, each in its own process group so that stopping it stops its descendants."""

    def __init__(self, result, prefix='', base_env=None):
        self.result, self.prefix, self.base_env = Path(result), prefix, base_env
        self.children = {}

    def launch(self, command, name, env=None, log=None):
        command = [str(part) for part in command]
        write_json(self.result / f'{self.prefix}{name}-command.json', command)
        base = os.environ if self.base_env is None else self.base_env
        log = Path(log or self.result / f'{self.prefix}{name}.out')
        self.children[name] = subprocess.Popen(command, env={**base, **(env or {})}, stdout=log.open('wb'),
                                               stderr=subprocess.STDOUT, start_new_session=True)
        return self.children[name]

    def check(self, allowed=()):
        """Exit if the campaign asked to stop; fail if a required child exited."""
        if (self.result / STOP_FILE).exists():
            raise SystemExit(STOP_EXIT_CODE)
        for name, process in self.children.items():
            if name not in allowed and process.poll() is not None:
                raise RuntimeError(f'{name} exited with {process.returncode}; inspect its log')

    def wait_for_file(self, name, timeout=900):
        deadline = time.monotonic() + timeout
        while not (self.result / name).exists():
            self.check()
            if time.monotonic() > deadline:
                raise TimeoutError(f'Readiness timed out for {name}')
            time.sleep(2)
        return json.loads((self.result / name).read_text())

    def run_until_exit(self, name):
        """Wait for one child while checking that the others stay alive."""
        while self.children[name].poll() is None:
            self.check(allowed=(name,))
            time.sleep(3)
        return self.children[name].returncode

    def stop_all(self, grace=0, timeout=8):
        """Allow `grace` seconds for a clean exit, then terminate, then kill."""
        processes = list(self.children.values())
        deadline = time.monotonic() + grace
        while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
            time.sleep(1)
        for process in reversed(processes):
            if process.poll() is None:
                _signal_group(process, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _signal_group(process, signal.SIGKILL)
                process.wait()


def _signal_group(process, number):
    try:
        os.killpg(process.pid, number)
    except ProcessLookupError:
        pass


class Allocation:
    """One two-node Slurm allocation: records, one step per role, and shutdown.

    The learner's exit status becomes the allocation's status. The generation
    role serves the learner, so its early exit is a failure.
    """

    def __init__(self, framework):
        self.site = site()
        self.job = os.environ['SLURM_JOB_ID']
        self.result = ROOT / 'results' / f'{framework}-{self.job}'

    def start(self, **details):
        self.result.mkdir(parents=True, exist_ok=False)
        nodes = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']],
                                        text=True).split()
        expected = [self.site['generation_node'], self.site['trainer_node'],
                    *([self.site['teacher_node']] if 'teacher_node' in self.site else [])]
        if sorted(nodes) != sorted(expected):  # Slurm lists nodes in its own order; `srun -w` pins each role.
            raise RuntimeError(f'Allocated nodes {nodes} differ from the site file {expected}')
        write_json(ROOT / 'active-run.json', {'job_id': self.job, 'result': str(self.result), 'nodes': nodes})
        write_json(self.result / 'start.json', {'time': time.time(), 'job_id': self.job, 'nodes': nodes, **details})
        (self.result / 'slurm-start.txt').write_bytes(subprocess.check_output(['scontrol', 'show', 'job', self.job]))
        entries = sorted(p.name for p in ROOT.iterdir() if p.name not in NOT_ARCHIVED)
        subprocess.run(['tar', '--exclude=__pycache__', '--exclude=.git', '--exclude=.venv', '-czf',
                        str(self.result / 'launch-source.tar.gz'), '-C', str(ROOT), *entries], check=True)

    def run(self, steps, env, before=None):
        """Run `steps` ({role: srun command}) until the learner exits, then stop everything."""
        exit_on_signals()
        roles = Supervisor(self.result)
        code = 1
        try:
            if before:
                before()
            for role, command in steps.items():
                roles.launch(command, role, env=env, log=self.result / f'{role}-driver.out')
            while roles.children['learner'].poll() is None:
                if roles.children['generation'].poll() is not None:
                    raise RuntimeError(f'Generation exited early with {roles.children["generation"].returncode}')
                time.sleep(5)
            code = roles.children['learner'].returncode
        finally:
            (self.result / STOP_FILE).touch()
            roles.stop_all(grace=40, timeout=20)
            (self.result / 'exit-code.txt').write_text(f'{code}\n')
            write_json(self.result / 'end.json', {'time': time.time(), 'exit_code': code})
        return code
