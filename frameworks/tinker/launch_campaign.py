"""Run an account-verified teacher → MOPD campaign, keeping all remote weights remote."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--credential-label', required=True)
    parser.add_argument('--expected-email', required=True)
    parser.add_argument('--expected-org', required=True)
    args = parser.parse_args()
    # Read a literal assignment without executing the user's shell profile or
    # printing any credential. Hyphenated labels are supported here explicitly.
    pattern = re.compile(r'^\s*(?:export\s+)?' + re.escape(args.credential_label) + r'\s*=\s*(.*)$')
    values = [shlex.split(m.group(1), comments=True)
              for line in (Path.home()/'.zprofile').read_text().splitlines()
              if (m := pattern.match(line))]
    if len(values) != 1 or len(values[0]) != 1:
        raise RuntimeError('Expected exactly one literal credential assignment')
    key = values[0][0]
    import httpx
    from tinker.cli.auth_api import TinkerAuthApi
    with httpx.Client(timeout=30) as client:
        identity = TinkerAuthApi(client).get_self_api_key(key)
    email = identity.details.user_details.email
    organization = identity.details.org_details.name
    if email != args.expected_email or organization != args.expected_org:
        raise RuntimeError(f'Account guard rejected {email} / {organization}')
    os.environ['TINKER_API_KEY'] = key
    os.environ.pop('TINKER_CREDENTIAL_CMD', None)
    os.environ['HF_HUB_OFFLINE'] = '1'  # Reuse the already cached tokenizer only.
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    snapshot = root/'source-repo'
    shutil.copytree(repo/'frameworks/tinker', snapshot/'frameworks/tinker',
                    ignore=shutil.ignore_patterns('__pycache__'))
    scorer = snapshot/'frameworks/slime/campaign/scoring.py'
    scorer.parent.mkdir(parents=True)
    shutil.copyfile(repo/'frameworks/slime/campaign/scoring.py', scorer)
    shutil.copytree(repo/'data', snapshot/'data')
    shutil.copytree(repo/'data', root/'research/data')
    shutil.copyfile(repo/'frameworks/tinker/README.md', root/'protocol.md')
    (root/'account.json').write_text(json.dumps({
        'email': email, 'organization': organization, 'key_name': identity.name,
        'credential_label': args.credential_label, 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'account_guard_passed': True}, indent=2))
    (root/'JOURNAL.md').write_text(
        '# SemiAnalysis Tinker campaign\n\n'
        f'Account verified before any model creation: {email} / {organization}. '
        'The personal-account campaign remains separate and unchanged.\n\n'
        'Repeat the corrected recipe: baseline; independent 40-update teachers; '
        'continue unmet provisional goals to at most 80 updates; choose frozen '
        'teachers; fresh-base 40-update, 128-response/update MOPD student. '
        'Goals: Countdown 50%, graph 90%; original teacher-quality matching remains '
        'unverified. Teacher multi-response seeds stay unset. No optimizer mutation '
        'retries or model-weight downloads. Sources are snapshotted before execution.\n')
    scripts = snapshot/'frameworks/tinker'
    lock = threading.Lock()

    def record(event, **fields):
        row = {'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}
        with lock:
            with (root/'campaign-events.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)

    def status(stage, **fields):
        temp = root/'status.tmp'
        temp.write_text(json.dumps({'stage': stage, **fields}, indent=2))
        temp.replace(root/'status.json')

    def execute(name, script, *arguments):
        command = [sys.executable, '-u', str(scripts/script), *map(str, arguments)]
        record('process_start', name=name, command=command)
        with (root/f'{name}-console.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record('process_exit', name=name, exit_code=result.returncode)
        if result.returncode:
            raise RuntimeError(f'{name} exited {result.returncode}; see retained console log')

    def teachers(domain):
        name = 'teacher-'+domain.replace('_', '-')
        initial = root/(name+'-v2')
        execute(name+'-v2', 'run.py', 'teacher', '--domain', domain,
                '--output', initial, '--skip-baseline')
        evaluations = [json.loads(x) for x in (initial/'evaluations.jsonl').read_text().splitlines()]
        target = {'countdown': 0.5, 'graph_color': 0.9}[domain]
        if not any(e[domain]['accuracy'] >= target for e in evaluations):
            checkpoints = [json.loads(x) for x in (initial/'checkpoints.jsonl').read_text().splitlines()]
            checkpoint = next(c['state_path'] for c in checkpoints if c['step'] == 40)
            execute(name+'-extension', 'run.py', 'teacher', '--domain', domain,
                    '--output', root/(name+'-extension'), '--checkpoint', checkpoint,
                    '--start-step', 40, '--steps', 80, '--target-score', target, '--skip-baseline')

    record('account_verified', email=email, organization=organization)
    try:
        status('baseline', pid=os.getpid(), account=email, organization=organization)
        execute('baseline', 'run.py', 'eval', '--output', root/'baseline')
        status('teachers', pid=os.getpid(), account=email, organization=organization)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(teachers, domain) for domain in ('countdown', 'graph_color')]
            failures = []
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    failures.append(str(error))
            if failures:
                raise RuntimeError('; '.join(failures))
        execute('teacher-selection', 'select_teachers.py', root)
        status('student', pid=os.getpid(), account=email, organization=organization)
        execute('student', 'run.py', 'mopd', '--teachers', root/'teachers.json', '--output', root/'student')
        execute('audit', 'audit.py', root)
        execute('metadata', 'collect_metadata.py', root)
        execute('analysis', 'analyze.py', root)
        execute('plot', 'plot.py', root)
        record('complete')
        status('complete', account=email, organization=organization, audit_passed=True)
        with (root/'JOURNAL.md').open('a') as stream:
            stream.write('\nCampaign completed; see campaign-events.jsonl for all process exit statuses, '
                         'REPORT.md for results, and final-audit.json for verification.\n')
    except BaseException:
        (root/'campaign-error.txt').write_text(traceback.format_exc())
        record('failed')
        status('failed', account=email, organization=organization, error_file='campaign-error.txt')
        raise


if __name__ == '__main__':
    main()
