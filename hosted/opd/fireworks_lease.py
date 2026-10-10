"""Provision, hold, and release the Fireworks resources that `--backend fireworks` runs attach to.

A lease is one dedicated LoRA trainer plus one sampling deployment for Qwen3.6-35B-A3B,
billed per GPU-hour while it exists. For OPD/MOPD, each `--teacher DOMAIN=MODEL_ID` (an uploaded
Fireworks model) also gets its own one-GPU deployment, and the lease writes `teachers.json`
pointing at them. Run the scripts one at a time against it (one LoRA session per trainer),
then release it:

    python hosted/opd/fireworks_lease.py up   --root /abs/lease --budget-usd 400 &   # holds the lease
    python hosted/opd/fireworks_lease.py up   --root /abs/lease --budget-usd 400 \
        --teacher caesar_cipher=qwen3p6-35b-a3b-caesar-cipher-sft-tinker &          # with a teacher
    FIREWORKS_LEASE=/abs/lease python hosted/opd/sft.py train --backend fireworks ...
    python hosted/opd/fireworks_lease.py down --root /abs/lease                      # deletes both

The independent watchdog of hosted/fireworks/lifecycle.py starts before anything is created.
It deletes the trainer and deployment when the conservative spend estimate reaches the budget
minus a cleanup reserve, when `down` is called, or when the `up` process dies, and it keeps
retrying until their release is observed. The estimate is client-side (elapsed time x
--rate-usd-hour), not an invoice or a provider-enforced cap.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent/'fireworks'))
from backend import Fireworks, service_config  # noqa: E402
from lifecycle import CLEANUP_RESERVE_USD, check, read_json, verify_account, write  # noqa: E402
from shared.credentials import api_key, local_config  # noqa: E402

GPU_USD_HOUR = 13  # Public on-demand B200 price.
TEACHER_GPU = 'NVIDIA_B200_180GB'
TEACHER_DEPLOYMENT_PREFIX = 'fireworks-deployment://'  # teachers.json paths that backend.Fireworks samples from.


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(root, event, **fields):
    row = {'utc': utc(), 'event': event, **fields}
    with (root/'lease-events.jsonl').open('a') as f:
        f.write(json.dumps(row, default=str) + '\n')
    print(json.dumps(row, default=str), flush=True)


BILLING_EVERY_SECONDS = 600


def billing(root, account, label):
    """Fireworks' own billed usage for today (UTC, aggregated daily, lagging real time), appended to billing.jsonl."""
    import httpx
    today = datetime.now(timezone.utc).date()
    params = {'startTime': f'{today}T00:00:00Z', 'endTime': f'{today.fromordinal(today.toordinal() + 1)}T00:00:00Z'}
    try:
        response = httpx.get(f'https://api.fireworks.ai/v1/accounts/{account}/billing/summary', params=params,
                             headers={'Authorization': 'Bearer ' + os.environ['FIREWORKS_API_KEY']}, timeout=30)
        response.raise_for_status()
        items = response.json().get('lineItems', [])
        cost = lambda i: int(i['totalCost'].get('units', 0)) + i['totalCost'].get('nanos', 0) / 1e9
        row = {'utc': utc(), 'label': label, 'day': str(today), 'total_usd': round(sum(cost(i) for i in items), 4),
               'items': [{'category': i.get('category'), 'gpu': i.get('groupingValue'), 'quantity': i.get('quantity'),
                          'usd': round(cost(i), 4)} for i in items]}
    except Exception as error:  # Billing is a record, never a reason to stop the lease.
        row = {'utc': utc(), 'label': label, 'error': str(error)[:300]}
    with (root/'billing.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    return row


def up(root, account, budget, rate, teachers, sampler_replicas, sampler_shape=None, teacher_shape=None,
       teachers_only=False, hot_load_transition='SYNC'):
    root.mkdir(parents=True, exist_ok=False)
    run_id = 'opd-lease-' + datetime.now(timezone.utc).strftime('%m%d%H%M%S')
    teacher_deployments = {d: f'{run_id}-teacher-{d.split("_")[0]}' for d in teachers}
    resources = {'account': account, 'teachers_only': teachers_only,
                 'trainer_id': None if teachers_only else run_id + '-train',
                 'deployment_id': None if teachers_only else run_id + '-sample',
                 'teacher_models': teachers, 'teacher_deployments': teacher_deployments,
                 'sampler_replicas': sampler_replicas, 'sampler_shape': sampler_shape, 'teacher_shape': teacher_shape,
                 'hot_load_transition_type': hot_load_transition,
                 'extra_deployments': list(teacher_deployments.values()),  # The watchdog deletes these too.
                 'budget_start_unix': time.time(), 'prior_spend_estimate_usd': 0, 'budget_usd': budget,
                 'stop_estimate_usd': budget - CLEANUP_RESERVE_USD, 'conservative_usd_hour': rate,
                 'controller_pid': os.getpid(), 'training_shape': Fireworks.SHAPE, 'created_utc': utc()}
    write(root/'resources.json', resources)
    write(root/'account.json', verify_account(account))
    baseline = billing(root, account, 'baseline')  # Today's billed total before this lease.
    log(root, 'billing_baseline', total_usd=baseline.get('total_usd'))
    with (root/'watchdog.log').open('w') as out:  # Its own session, so it outlives this process.
        watchdog = subprocess.Popen([sys.executable, str(HERE.parent/'fireworks/lifecycle.py'), str(root),
                                     '--pid', str(os.getpid())], stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
    write(root/'watchdog.json', {'pid': watchdog.pid})
    for _ in range(60):
        if (root/'watchdog-ready.json').exists():
            break
        if watchdog.poll() is not None:
            raise RuntimeError('The watchdog failed to start; see watchdog.log')
        time.sleep(1)
    else:
        raise RuntimeError('The watchdog did not become ready')
    log(root, 'watchdog_ready', pid=watchdog.pid, budget_usd=budget, stop_estimate_usd=resources['stop_estimate_usd'],
        rate_usd_hour=rate)

    from fireworks.training.sdk import FiretitanServiceClient, TrainerJobConfig, TrainerJobManager
    try:
        start = time.monotonic()
        log(root, 'provisioning', trainer_id=resources['trainer_id'], deployment_id=resources['deployment_id'])
        if not teachers_only:  # A teacher-only lease serves teachers for a serverless student.
            TrainerJobManager(api_key=os.environ['FIREWORKS_API_KEY']).create(TrainerJobConfig(
                base_model=Fireworks.MODELS['Qwen/Qwen3.6-35B-A3B'], lora_rank=Fireworks.MAX_LORA_RANK,
                learning_rate=1e-5, training_shape_ref=Fireworks.SHAPE, trainer_replica_count=1,
                requested_job_id=resources['trainer_id'], inactivity_timeout='1800s', display_name=root.name))
            log(root, 'trainer_requested', seconds=time.monotonic() - start)
        request_teachers(root, resources)
        check(root)
        wait_ready(root, resources, start)
        last_billing = time.monotonic()
        while not (root/'RELEASE').exists():  # Hold until `down`; the watchdog may also stop the lease.
            check(root)
            if time.monotonic() - last_billing >= BILLING_EVERY_SECONDS:
                row = billing(root, account, 'during')
                log(root, 'billing', total_usd=row.get('total_usd'),
                    lease_usd=round(row['total_usd'] - baseline['total_usd'], 2) if 'total_usd' in row and 'total_usd' in baseline else None)
                last_billing = time.monotonic()
            time.sleep(15)
        log(root, 'release_requested')
    finally:
        (root/'TRAINING_FINISHED').write_text(utc())  # The watchdog deletes both resources and verifies it.
        for _ in range(240):
            if (root/'cleanup.json').exists():
                log(root, 'released', cleanup=read_json(root/'cleanup.json').get('verified_released'))
                break
            time.sleep(5)
        else:
            log(root, 'release_not_yet_verified', note='The watchdog keeps retrying; check lifecycle.jsonl')
        billing(root, account, 'after_release')  # Billing lags; `fireworks_lease.py billing` records it again later.


def request_teachers(root, resources):
    """One single-GPU deployment per uploaded teacher model, serving it as-is (no hot-load)."""
    from fireworks.training.sdk.deployment import DeploymentConfig, DeploymentManager
    manager = DeploymentManager(api_key=os.environ['FIREWORKS_API_KEY'])
    for domain, model_id in resources['teacher_models'].items():
        deployment_id = resources['teacher_deployments'][domain]
        manager.create_or_get(DeploymentConfig(
            deployment_id=deployment_id, base_model=f'accounts/{resources["account"]}/models/{model_id}',
            min_replica_count=1, max_replica_count=1, accelerator_type=TEACHER_GPU, enable_hot_load=False,
            hot_load_bucket_type=None, disable_speculative_decoding=True,
            deployment_shape=resources.get('teacher_shape'),  # A validated shape; shapeless only without one.
            accept_shapeless_risk=not resources.get('teacher_shape'),
            description=f'OPD teacher: {domain}'))
        log(root, 'teacher_requested', domain=domain, model=model_id, deployment_id=deployment_id)


def wait_ready(root, resources, start=None):
    """Attach and provision eagerly: wait for the trainer (capacity, then healthy) and create the deployment.

    The SDK otherwise provisions on first use, which would count provisioning inside the first run's timings.
    """
    from fireworks.training.sdk import FiretitanServiceClient
    start = time.monotonic() if start is None else start
    if not resources.get('teachers_only'):
        service = FiretitanServiceClient.from_firetitan_config(**service_config(resources, 'opd-lease'))
        try:
            service._ensure_managed_handle()  # Private in fireworks-ai 1.2.14: the eager form of first-use provisioning.
        finally:
            service.close()
    if resources.get('teacher_deployments'):
        from fireworks.training.sdk.deployment import DeploymentManager
        manager = DeploymentManager(api_key=os.environ['FIREWORKS_API_KEY'])
        paths = {}
        for domain, deployment_id in resources['teacher_deployments'].items():
            manager.wait_for_ready(deployment_id, timeout_s=3600)
            paths[domain] = f'{TEACHER_DEPLOYMENT_PREFIX}accounts/{resources["account"]}/deployments/{deployment_id}'
            log(root, 'teacher_ready', domain=domain, deployment_id=deployment_id, seconds=time.monotonic() - start)
        write(root/'teachers.json', paths)
    write(root/'lease.json', {'ready': True, 'ready_utc': utc(), 'provisioning_seconds': time.monotonic() - start,
                              **{k: resources[k] for k in ('account', 'trainer_id', 'deployment_id')},
                              'teacher_deployments': resources.get('teacher_deployments', {})})
    log(root, 'ready', provisioning_seconds=time.monotonic() - start)


def drop_teacher(root, domain):
    """Delete one teacher's deployment mid-lease (e.g. geometry, before OPD) and lower the spend-estimate rate."""
    import httpx
    resources = read_json(root/'resources.json')
    deployment_id = resources['teacher_deployments'][domain]
    response = httpx.delete(f'https://api.fireworks.ai/v1/accounts/{resources["account"]}/deployments/{deployment_id}',
                            params={'ignoreChecks': 'true'}, headers={'Authorization': 'Bearer ' + os.environ['FIREWORKS_API_KEY']},
                            timeout=30)
    response.raise_for_status()
    resources['conservative_usd_hour'] -= GPU_USD_HOUR  # The watchdog re-reads resources.json.
    resources.setdefault('dropped_teachers', {})[domain] = utc()
    write(root/'resources.json', resources)
    log(root, 'teacher_dropped', domain=domain, deployment_id=deployment_id, rate_usd_hour=resources['conservative_usd_hour'])


def down(root):
    (root/'RELEASE').write_text(utc())
    for _ in range(360):
        if (root/'cleanup.json').exists():
            print(json.dumps(read_json(root/'cleanup.json'), indent=2, default=str)[:2000])
            return
        time.sleep(5)
    raise SystemExit('Release not verified within 30 minutes; see lifecycle.jsonl')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('up', 'wait', 'down', 'billing', 'drop-teacher'))
    p.add_argument('--domain', help='drop-teacher: the teacher to delete')
    p.add_argument('--root', type=Path, required=True, help='The lease directory (new for up)')
    p.add_argument('--account', help='Default: config/fireworks.local.json, then $FIREWORKS_ACCOUNT_ID')
    p.add_argument('--budget-usd', type=float, help='Required for up: total spend estimate at which to stop')
    p.add_argument('--teacher', action='append', default=[], metavar='DOMAIN=MODEL_ID',
                   help='An uploaded teacher model to deploy for OPD/MOPD (repeatable)')
    p.add_argument('--sampler-replicas', type=int, default=1, help='Replicas of the student sampling deployment')
    p.add_argument('--sampler-shape', help="Deployment shape version for the sampler (default: the training shape's paired 2-GPU shape)")
    p.add_argument('--sampler-gpus', type=int, default=2, help='GPUs per sampler replica, for the spend estimate')
    p.add_argument('--hot-load-transition', choices=('SYNC', 'ASYNC'), default='SYNC',
                   help='How the sampler swaps weights with requests in flight: SYNC lets them finish on the old '
                        'weights first; ASYNC (the Fireworks default) pauses and resumes them on the new weights')
    p.add_argument('--teacher-shape', help='Validated deployment shape version for the teachers (default: shapeless)')
    p.add_argument('--teachers-only', action='store_true',
                   help='Only the teacher deployments, for a serverless student (no trainer or sampling deployment)')
    p.add_argument('--rate-usd-hour', type=float, help='Spend estimate rate; default: $13 per GPU of the lease')
    args = p.parse_args()
    teachers = dict(t.split('=', 1) for t in args.teacher)
    # Trainer 4 GPUs (the training shape), the sampler's GPUs per replica, teachers 1 each.
    trainer_and_sampler = 0 if args.teachers_only else 4 + args.sampler_gpus * args.sampler_replicas
    rate = args.rate_usd_hour or GPU_USD_HOUR * (trainer_and_sampler + len(teachers))
    root = args.root.resolve()
    if args.action == 'down':
        return down(root)
    settings = local_config('fireworks')
    if args.action == 'drop-teacher':
        os.environ['FIREWORKS_API_KEY'] = api_key('FIREWORKS_API_KEY', settings.get('api_key_profile_label'))
        return drop_teacher(root, args.domain)
    if args.action == 'billing':  # Record billed usage again, e.g. hours after release once it has settled.
        os.environ['FIREWORKS_API_KEY'] = api_key('FIREWORKS_API_KEY', settings.get('api_key_profile_label'))
        rows = [json.loads(l) for l in (root/'billing.jsonl').read_text().splitlines()]
        row = billing(root, read_json(root/'resources.json')['account'], 'later')
        base = next(r for r in rows if r['label'] == 'baseline')
        return print(json.dumps({'today_usd': row.get('total_usd'), 'lease_usd': round(row['total_usd'] - base['total_usd'], 2)}))
    if args.action == 'wait':  # Eagerly provision a lease whose `up` returned before its resources were ready.
        os.environ['FIREWORKS_API_KEY'] = api_key('FIREWORKS_API_KEY', settings.get('api_key_profile_label'))
        resources = read_json(root/'resources.json')
        return wait_ready(root, resources, start=time.monotonic() - (time.time() - resources['budget_start_unix']))
    account = args.account or settings.get('account') or os.environ.get('FIREWORKS_ACCOUNT_ID')
    if not account: p.error('Set --account, config/fireworks.local.json, or $FIREWORKS_ACCOUNT_ID')
    if not args.budget_usd or args.budget_usd <= CLEANUP_RESERVE_USD: p.error(f'--budget-usd must exceed {CLEANUP_RESERVE_USD}')
    os.environ['FIREWORKS_API_KEY'] = api_key('FIREWORKS_API_KEY', settings.get('api_key_profile_label'))
    up(root, account, args.budget_usd, rate, teachers, args.sampler_replicas, args.sampler_shape, args.teacher_shape,
       args.teachers_only, args.hot_load_transition)


if __name__ == '__main__':
    main()
