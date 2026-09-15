"""Route puzzle domains through Slime's native OPD hooks and retain evidence."""

import copy
import json
import math
import os
from pathlib import Path
import time

from scoring import score
from slime.rollout import on_policy_distillation as opd


def record(name, row):
    target = Path(os.environ['MOPD_RESULT']) / name
    with target.open('a') as stream:
        stream.write(json.dumps({'time': time.time(), **row}, allow_nan=False) + '\n')


def stamp(args, sample, *, rollout_id=None, evaluation=False):
    sample.metadata = dict(sample.metadata or {})
    sample.metadata['campaign_rollout_id'] = rollout_id
    sample.metadata['campaign_evaluation'] = evaluation
    return sample


async def reward(args, sample, **kwargs):
    domain = sample.metadata['domain']
    if domain not in ('countdown', 'graph_color'):
        raise ValueError(f'Unexpected teacher domain: {domain}')
    task_score = score(sample.response, sample.label)
    sample.metadata['task_score'] = task_score
    if sample.metadata['campaign_evaluation']:
        return {'task_score': task_score}
    selected = copy.copy(args)
    selected.rm_url = json.loads(os.environ['MOPD_TEACHER_URLS'])[domain]
    started = time.monotonic()
    response = await opd.reward_func(selected, sample, **kwargs)
    assert response['meta_info'].get('completion_tokens', 0) == 0
    record('teacher-requests.jsonl', {'domain': domain, 'index': sample.index,
        'seconds': time.monotonic()-started, 'input_tokens': len(sample.tokens),
        'response_tokens': sample.response_length, 'versions': sample.weight_versions})
    return {'task_score': task_score, 'opd': response}


def postprocess(args, samples):
    # Copy args locally so concurrent requests never mutate the shared teacher URL or key.
    selected = copy.copy(args)
    selected.reward_key = 'opd'
    raw, processed = opd.post_process_rewards(selected, samples)
    assert raw == processed == [0.0] * len(samples)
    for sample in samples:
        scores = sample.teacher_log_probs
        assert len(scores) == sample.response_length and scores.isfinite().all().item()
    # Task scores are diagnostic only. Native OPD receives zero scalar task reward.
    return [s.metadata['task_score'] for s in samples], processed


def sample_record(sample):
    return {'index': sample.index, 'domain': sample.metadata['domain'], 'label': sample.label,
        'response': sample.response, 'response_length': sample.response_length,
        'tokens': sample.tokens, 'rollout_log_probs': sample.rollout_log_probs,
        'weight_versions': sample.weight_versions, 'task_score': sample.metadata['task_score'],
        'status': sample.status.name, 'metadata': sample.metadata}


def log_train(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    assert len(samples) == 128
    lags = []
    for sample in samples:
        # Native distributed broadcast starts at version one for the initial base weights.
        versions = [int(v) for v in sample.weight_versions]
        assert versions, 'SGLang did not return a policy version'
        sample_lags = [rollout_id + 2 - v for v in versions]
        assert all(0 <= lag <= 2 for lag in sample_lags), (rollout_id, versions)
        lags.extend(sample_lags)
        record('train-samples.jsonl', {'rollout_id': rollout_id, **sample_record(sample)})
    record('rollouts.jsonl', {'rollout_id': rollout_id, 'samples': len(samples), 'seconds': rollout_time,
        'lag_mean': sum(lags)/len(lags), 'lag_max': max(lags), 'native_metrics': rollout_extra_metrics})
    return False


def log_eval(rollout_id, args, data, extra_metrics):
    expected_version = 1 if rollout_id == -1 else rollout_id + 2
    for domain, payload in data.items():
        samples = payload['samples']
        assert len(samples) == 512
        versions = sorted({int(v) for s in samples for v in s.weight_versions})
        assert versions == [expected_version], (rollout_id, domain, versions)
        for sample in samples:
            record('eval-samples.jsonl', {'rollout_id': rollout_id, **sample_record(sample)})
        record('evaluations.jsonl', {'rollout_id': rollout_id, 'updates': rollout_id+1,
            'domain': domain, 'samples': len(samples), 'versions': versions,
            'score': sum(s.metadata['task_score'] for s in samples)/len(samples),
            'truncated': sum(s.status.name == 'TRUNCATED' for s in samples)})
    return False
