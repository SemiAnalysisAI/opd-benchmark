"""Run the hosted Tinker MOPD campaign.

Full campaign: evaluate the base model, train one GRPO teacher per domain
(stopping at its goal), select the teachers, train a fresh MOPD student, and
write the report. With --teachers sft, each teacher is instead a LoRA SFT on
the released traces of the frozen recipe.TEACHERS (hosted/opd/sft.py), and the base model
and teachers get the 100-problem x 3-sample benchmark. --sft-config JSON changes
sft.py settings, for every stage or per domain (see sft_arguments). With --teachers-only, stop
once the teachers are selected. With --teacher-campaign, reuse that campaign's
selected teachers and train only the student.

The key comes from $TINKER_API_KEY, or from the `~/.zprofile` label in
config/tinker.local.json. If that file sets expected_email and expected_org,
any other account is refused. Every stage runs from a snapshot of the code and
data in <root>/source-repo; model weights stay remote.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from shared.credentials import api_key, local_config  # noqa: E402

# Operational own-domain dev accuracy goals (greedy, full dev split). They are set just below
# the research GRPO teachers' held-out scores (caesar_cipher 73.3%, simple_geometry 99.8%;
# research/mopd-2026-09-25/README.md), not a verified match of the frozen recipe.TEACHERS.
# A teacher stops at its first scheduled evaluation meeting its goal, or after run.py's TEACHER steps.
TEACHER_GOALS = {'caesar_cipher': 0.70, 'simple_geometry': 0.95}
SFT_SCRIPT = '../opd/sft.py'  # --teachers sft runs hosted/opd/sft.py on its Tinker backend.


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def verify_account(settings):
    """Return the key and account.json record; with a configured guard, refuse any other account."""
    label = settings.get('api_key_profile_label')
    key = api_key('TINKER_API_KEY', label)
    import httpx
    from tinker.cli.auth_api import TinkerAuthApi
    with httpx.Client(timeout=30) as client:
        identity = TinkerAuthApi(client).get_self_api_key(key)
    email, organization = identity.details.user_details.email, identity.details.org_details.name
    expected = settings.get('expected_email'), settings.get('expected_org')
    guarded = any(expected)
    if guarded and (email, organization) != expected:
        raise RuntimeError(f'Account guard rejected {email} / {organization}')
    return key, {'email': email, 'organization': organization, 'key_name': identity.name,
                 'credential_label': label or 'TINKER_API_KEY', 'verified_utc': utc_now(),
                 'account_guard_passed': guarded}


def choose(rows, domain, goal):
    """First evaluated checkpoint meeting the goal; otherwise the best, earliest on ties."""
    qualifying = [r for r in rows if r['evaluation'][domain]['accuracy'] >= goal]
    if qualifying:
        chosen = min(qualifying, key=lambda r: r['step'])
    else:
        chosen = max(rows, key=lambda r: (r['evaluation'][domain]['accuracy'], -r['step']))
    return {**chosen, 'provisional_target': goal, 'target_met': bool(qualifying)}


def select_teachers(root):
    """Write teacher-selection.json and teachers.json from the completed teacher runs."""
    candidates = {d: [] for d in TEACHER_GOALS}
    for config_path in sorted(root.glob('*/config.json')):
        path = config_path.parent
        config = json.loads(config_path.read_text())
        if config['mode'] != 'teacher' or (path/'INVALIDATED.json').exists():
            continue
        if read_jsonl(path/'events.jsonl')[-1]['event'] != 'complete':
            raise RuntimeError(f'Teacher attempt is not complete: {path.name}')
        checkpoints = {r['step']: r for r in read_jsonl(path/'checkpoints.jsonl')}
        for evaluation in read_jsonl(path/'evaluations.jsonl'):
            if evaluation['step'] in checkpoints:
                candidates[config['domain']].append({'source_run': path.name, 'step': evaluation['step'],
                    'evaluation': evaluation, 'checkpoint': checkpoints[evaluation['step']]})
    missing = [d for d, rows in candidates.items() if not rows]
    if missing:
        raise RuntimeError(f'No saved, evaluated teacher for {", ".join(missing)}')
    selected = {d: choose(rows, d, TEACHER_GOALS[d]) for d, rows in candidates.items()}
    routes = {d: r['checkpoint']['sampler_path'] for d, r in selected.items()}
    assert len(set(routes.values())) == len(routes)
    report = {'rule': 'First scheduled checkpoint reaching provisional goal; otherwise highest own-domain dev score, earliest on ties.',
        'original_teacher_quality_match_verified': False,
        'target_source': 'Operational goals below the research GRPO teachers\' held-out scores; not a verified quality match.',
        'selected': selected}
    write_selection(root, report, routes)


def write_selection(root, report, routes):
    for filename, value in [('teacher-selection.json', report), ('teachers.json', routes)]:
        with (root/filename).open('x') as f:  # Never overwrite a selection.
            json.dump(value, f, indent=2)


def select_sft_teachers(root):
    """Each domain's SFT teacher is its single, final checkpoint."""
    selected = {}
    for d in TEACHER_GOALS:
        teacher = json.loads((root/sft_name(d)/'teacher.json').read_text())
        assert teacher['domain'] == d
        selected[d] = {'source_run': sft_name(d), 'step': teacher['steps'], 'benchmark': teacher['benchmark'],
                       'checkpoint': {'sampler_path': teacher['sampler_path'], 'state_path': teacher['state_path']}}
    routes = {d: r['checkpoint']['sampler_path'] for d, r in selected.items()}
    assert len(set(routes.values())) == len(routes)
    report = {'rule': 'SFT teachers: the final checkpoint of one epoch on the released traces; no selection.',
        'original_teacher_quality_match_verified': False, 'selected': selected}
    write_selection(root, report, routes)


def sft_name(domain):
    return 'teacher-' + domain.replace('_', '-')


def sft_arguments(settings, domain=None):
    """sft.py flags from an --sft-config object: {"learning_rate": 2e-4, "domains": {"caesar_cipher": {...}}}.

    Top-level settings apply to every stage (the baseline too); a domain's section applies to its teacher only.
    Keys are sft.py flag names with underscores; sft.py rejects unknown ones.
    """
    merged = {k: v for k, v in settings.items() if k != 'domains'}
    if domain:
        merged.update(settings.get('domains', {}).get(domain, {}))
    unknown = set(settings.get('domains', {})) - set(TEACHER_GOALS)
    if unknown:
        raise ValueError(f'Unknown domains in --sft-config: {sorted(unknown)}')
    return [x for k, v in merged.items() for x in ('--' + k.replace('_', '-'), str(v))]


class Campaign:
    def __init__(self, root):
        self.root = root
        self.scripts = root/'source-repo/hosted/tinker'
        self.lock = threading.Lock()  # Teachers train concurrently.
        self.sft_settings = json.loads((root/'sft-config.json').read_text()) if (root/'sft-config.json').exists() else {}

    def record(self, event, **fields):
        row = {'utc': utc_now(), 'event': event, **fields}
        with self.lock:
            with (self.root/'campaign-events.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)

    def status(self, stage, **fields):
        temp = self.root/'status.tmp'
        temp.write_text(json.dumps({'stage': stage, 'pid': os.getpid(), 'utc': utc_now(), **fields}, indent=2))
        temp.replace(self.root/'status.json')

    def execute(self, name, script, *arguments):
        """Run one snapshot script; its console output goes to <name>-console.log."""
        command = [sys.executable, '-u', str(self.scripts/script), *map(str, arguments)]
        self.record('process_start', name=name, command=command)
        with (self.root/f'{name}-console.log').open('w') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        self.record('process_exit', name=name, exit_code=code)
        if code:
            raise RuntimeError(f'{name} exited {code}; see retained console log')

    def teacher(self, domain):
        name = 'teacher-' + domain.replace('_', '-')
        self.execute(name, 'run.py', 'teacher', '--domain', domain, '--output', self.root/name,
                     '--target-score', TEACHER_GOALS[domain], '--skip-baseline')

    def sft_teacher(self, domain):
        self.execute(sft_name(domain), SFT_SCRIPT, 'train', '--domain', domain, '--output', self.root/sft_name(domain),
                     *sft_arguments(self.sft_settings, domain))

    def run(self, prior, pipeline_depth, method, teachers_only):
        if prior:
            self.record('reusing_teachers', teacher_campaign=str(prior))
        elif method == 'sft':
            # The base-model benchmark runs alongside the teachers.
            self.status('teachers')
            with ThreadPoolExecutor(max_workers=1 + len(TEACHER_GOALS)) as pool:
                baseline = pool.submit(self.execute, 'baseline', SFT_SCRIPT, 'bench', '--output', self.root/'baseline',
                                       *sft_arguments(self.sft_settings))
                list(pool.map(self.sft_teacher, TEACHER_GOALS))
                baseline.result()
            select_sft_teachers(self.root)
        else:
            self.status('baseline')
            self.execute('baseline', 'run.py', 'eval', '--output', self.root/'baseline')
            self.status('teachers')
            # Leaving the pool waits for both teachers, even when one fails.
            with ThreadPoolExecutor(max_workers=len(TEACHER_GOALS)) as pool:
                list(pool.map(self.teacher, TEACHER_GOALS))
            select_teachers(self.root)
        if teachers_only:
            self.record('complete', teachers_only=True)
            self.status('complete', teachers_only=True)
            return
        self.status('student')
        self.execute('student', 'run.py', 'mopd', '--teachers', self.root/'teachers.json',
                     '--output', self.root/'student', '--pipeline-depth', pipeline_depth)
        self.status('report')
        self.execute('report', 'report.py', self.root)
        self.record('complete')
        self.status('complete', audit_passed=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True, help='New campaign directory')
    p.add_argument('--config', type=Path, help='Account settings (default: config/tinker.local.json, if present)')
    p.add_argument('--teacher-campaign', type=Path, help='Reuse the selected teachers of this campaign')
    p.add_argument('--pipeline-depth', type=int, choices=(0, 1), default=1,
                   help='Student lookahead (default 1: policy lag at most recipe.MAX_POLICY_LAG)')
    p.add_argument('--teachers', choices=('grpo', 'sft'), default='grpo',
                   help='How to build the teachers: GRPO on the tasks (default), or SFT on the released traces')
    p.add_argument('--teachers-only', action='store_true', help='Stop after the teachers are selected')
    p.add_argument('--sft-config', type=Path, help='JSON of sft.py settings for --teachers sft (see sft_arguments)')
    args = p.parse_args()
    if args.sft_config and args.teachers != 'sft':
        p.error('--sft-config requires --teachers sft')
    sft_settings = json.loads(args.sft_config.read_text()) if args.sft_config else {}
    for domain in TEACHER_GOALS:
        sft_arguments(sft_settings, domain)  # Fail on a malformed file before the campaign starts.
    if args.teacher_campaign and (args.teachers_only or args.teachers != 'grpo'):
        p.error('--teacher-campaign reuses teachers; it takes neither --teachers nor --teachers-only')
    root = args.root.resolve()
    prior = args.teacher_campaign.resolve() if args.teacher_campaign else None
    if prior:
        selection = json.loads((prior/'teacher-selection.json').read_text())
        routes = json.loads((prior/'teachers.json').read_text())
        if routes != {d: v['checkpoint']['sampler_path'] for d, v in selection['selected'].items()}:
            raise RuntimeError('teachers.json does not match teacher-selection.json')
    key, account = verify_account(local_config('tinker', args.config))
    os.environ['TINKER_API_KEY'] = key  # This process and its stages only.
    os.environ.pop('TINKER_CREDENTIAL_CMD', None)
    os.environ['HF_HUB_OFFLINE'] = '1'  # Reuse the cached pinned tokenizer only.
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # Keep the snapshot unchanged.
    root.mkdir(parents=True, exist_ok=False)
    for package in ('hosted/tinker', 'hosted/opd', 'shared'):
        shutil.copytree(REPO/package, root/'source-repo'/package, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(REPO/'data', root/'source-repo/data')
    if prior:
        for name in ('teachers.json', 'teacher-selection.json'):
            shutil.copyfile(prior/name, root/name)
    (root/'account.json').write_text(json.dumps(account, indent=2))
    if args.teachers == 'sft':
        (root/'sft-config.json').write_text(json.dumps(sft_settings, indent=2))
    campaign = Campaign(root)
    campaign.record('account_verified', email=account['email'], organization=account['organization'])
    try:
        campaign.run(prior, args.pipeline_depth, args.teachers, args.teachers_only)
    except BaseException:
        (root/'campaign-error.txt').write_text(traceback.format_exc())
        campaign.record('failed')
        campaign.status('failed', error_file='campaign-error.txt')
        raise


if __name__ == '__main__':
    main()
