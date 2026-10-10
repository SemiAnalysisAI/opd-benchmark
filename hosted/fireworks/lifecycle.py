"""Credentials, the budget estimate, and the independent fail-closed watchdog.

`python lifecycle.py RUN_ROOT --pid CONTROLLER_PID` requests a stop when the spend estimate reaches the
run's threshold, training finishes, or the controller dies, then deletes this run's two named resources
(account and names from `resources.json`) until their release is observed.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse, json, os, sys, time

import httpx

REPO = Path(__file__).resolve().parents[2]  # The frozen `source-repo/` copy in a launched run.
sys.path.insert(0, str(REPO))  # Makes the shared package importable.

API_URL = 'https://api.fireworks.ai'
# Conservative client-side estimate, not an invoice or a provider cap: six B200s (the four-GPU
# trainer plus the rollout replica) at the public $13/hour, plus 50% headroom.
GPU_USD_HOUR, ALLOCATED_GPUS, HEADROOM = 13, 6, 1.5
RATE_USD_HOUR = GPU_USD_HOUR * ALLOCATED_GPUS * HEADROOM  # 117
DEFAULT_BUDGET_USD = 200
CLEANUP_RESERVE_USD = 25  # Kept back so shutdown can finish within the budget.
STOP_USD = DEFAULT_BUDGET_USD - CLEANUP_RESERVE_USD  # For a run that records no stop.


def api_key():
    # campaign.py resolves the key once (environment or ~/.zprofile label) and passes it on
    # to its frozen copy and the watchdog through the environment.
    if not os.environ.get('FIREWORKS_API_KEY'):
        raise RuntimeError('FIREWORKS_API_KEY is not set')
    return os.environ['FIREWORKS_API_KEY']


def headers():
    return {'Authorization': 'Bearer ' + api_key()}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str) + '\n')


def event(root, name, **fields):
    with (Path(root) / 'lifecycle.jsonl').open('a') as file:
        file.write(json.dumps(dict(utc=utc_now(), event=name, **fields), default=str) + '\n')


def resource_estimate(resources, now=None):
    """Prior attempts' spend plus allocation time since provisioning at the run's rate."""
    elapsed = max(0, (time.time() if now is None else now) - resources['budget_start_unix'])
    return (resources.get('prior_spend_estimate_usd', 0)
            + elapsed * resources.get('conservative_usd_hour', RATE_USD_HOUR) / 3600)


def verify_account(account):
    with httpx.Client(headers=headers(), timeout=30) as client:
        response = client.get(API_URL + '/v1/accounts')
        response.raise_for_status()
    assert any(x['name'] == 'accounts/' + account for x in response.json()['accounts']), f'No access to {account}'
    return {'account': account, 'verified_utc': utc_now()}


def paths(resources):
    """This run's named resources. `extra_deployments` (hosted/opd teacher deployments) is optional, and a
    teacher-only lease has no trainer or main deployment."""
    account = f'/v1/accounts/{resources["account"]}'
    named = {}
    if resources.get('trainer_id'):
        named['trainer'] = f'{account}/rlorTrainerJobs/{resources["trainer_id"]}'
    if resources.get('deployment_id'):
        named['deployment'] = f'{account}/deployments/{resources["deployment_id"]}'
    for deployment_id in resources.get('extra_deployments', []):
        named['deployment:' + deployment_id] = f'{account}/deployments/{deployment_id}'
    return named


def snapshot(root, resources):
    result = {}
    with httpx.Client(headers=headers(), timeout=20) as client:
        for kind, path in paths(resources).items():
            response = client.get(API_URL + path)
            result[kind] = {'http_status': response.status_code, 'data': response.json() if response.content else {}}
    with (Path(root) / 'resource-metrics.jsonl').open('a') as file:
        file.write(json.dumps({'utc': utc_now(), 'resources': result}) + '\n')
    return result


def released(state):
    trainer = state.get('trainer', {'http_status': 404, 'data': {}})

    def deployment_released(deployment):
        data = deployment['data']
        # A scale-down must be observed, not only requested.
        scaled_to_zero = deployment['http_status'] == 200 and (
            data.get('minReplicaCount', 0) == 0 and data.get('maxReplicaCount', -1) == 0
            and data.get('currentReplicaCount', -1) == 0)
        return deployment['http_status'] == 404 or data.get('state') == 'DELETED' or scaled_to_zero

    return ((trainer['http_status'] == 404 or trainer['data'].get('state') in ('DELETED', 'JOB_STATE_DELETED'))
            and all(deployment_released(v) for k, v in state.items() if k.startswith('deployment')))


def cleanup(root, resources):
    """Delete only this run's named trainer and deployment; checkpoints remain server-side."""
    with httpx.Client(headers=headers(), timeout=30) as client:
        for kind, path in paths(resources).items():
            # The run owns its deployment; recent rollout traffic must not block shutdown (as in the SDK).
            response = client.delete(API_URL + path, params={'ignoreChecks': 'true'} if kind.startswith('deployment') else None)
            event(root, 'cleanup_request', resource=kind, status=response.status_code,
                  response=response.json() if response.content else {})


def check(root):
    """Raise if a stop was requested or the estimate reached the stop threshold."""
    root = Path(root)
    if (root / 'STOP').exists():
        raise RuntimeError('Budget/lifecycle watchdog requested stop')
    if (root / 'resources.json').exists():
        resources = read_json(root / 'resources.json')
        if resource_estimate(resources) >= resources.get('stop_estimate_usd', STOP_USD):
            (root / 'STOP').write_text('Conservative spend reserve reached\n')
            raise RuntimeError('Conservative spend reserve reached')


def watchdog(root, pid):
    root = Path(root)
    stop_at = None
    while True:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        resources = read_json(root / 'resources.json')  # Re-read, so budget corrections apply.
        cost = resource_estimate(resources)
        must_stop = (cost >= resources.get('stop_estimate_usd', STOP_USD) or not alive
                     or (root / 'TRAINING_FINISHED').exists() or (root / 'STOP').exists())
        if must_stop:
            if stop_at is None:
                stop_at = time.time()
                (root / 'STOP').write_text('Watchdog stop/cleanup requested\n')
                event(root, 'stop_requested', estimated_usd=cost, parent_alive=alive)
            # Give the controller 30 s to finish its current request and save state.
            if not alive or (root / 'TRAINING_FINISHED').exists() or time.time() - stop_at >= 30:
                try:
                    cleanup(root, resources)
                except Exception as error:
                    event(root, 'cleanup_error', error=str(error))
        try:
            state = snapshot(root, resources)
            if not (root / 'watchdog-ready.json').exists():
                write(root / 'watchdog-ready.json', {'pid': os.getpid(), 'utc': utc_now()})
            event(root, 'budget_observation', conservative_estimate_usd=cost,
                  rate_usd_hour=resources.get('conservative_usd_hour', RATE_USD_HOUR))
            if must_stop and released(state):
                write(root / 'cleanup.json', {'verified_released': True, 'utc': utc_now(),
                                              'conservative_estimate_usd': cost,
                                              'billing': 'Estimate only; not an invoice', 'resources': state})
                try:
                    from report import report
                    report(root)
                except Exception as error:
                    event(root, 'audit_error', error=str(error))
                return
        except Exception as error:
            event(root, 'observation_error', error=str(error))
        time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    watchdog(args.root, args.pid)
