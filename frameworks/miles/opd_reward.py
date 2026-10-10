"""Miles reward hook: score with the routed frozen teacher, retrying transient transport errors.

Evaluation samples are scored by the shared task verifier instead. Training samples also store the
verifier score in `metadata["raw_reward"]`, so Miles logs it as `rollout/raw_reward` instead of the
teacher reward's constant 0. The OPD loss ignores it. Each teacher attempt is appended to
`teacher-latency-<pid>.jsonl`. `log_rollout` appends each training batch's verifier score per task to
`train-reward.jsonl` and one row per trained sample (task, score, lengths, status) to `train-samples.jsonl`.
"""
import asyncio
import json
import logging
import os
from pathlib import Path
import time

import httpx

from miles.rollout.on_policy_distillation import reward_func as teacher_reward
from shared.scoring import score

ATTEMPTS = 3
FIRST_RETRY_SECONDS = 0.25
_stream = None
logger = logging.getLogger(__name__)


def _record_attempt(sample, attempt, started, total_started, success, error):
    global _stream
    if _stream is None:
        path = Path(os.environ['CAMPAIGN_RESULT']) / f'teacher-latency-{os.getpid()}.jsonl'
        _stream = path.open('a', buffering=8192)
    _stream.write(json.dumps({'time_unix': time.time(), 'seconds': time.monotonic() - started,
                              'total_seconds': time.monotonic() - total_started,
                              'success': success, 'attempt': attempt, 'error_type': error,
                              'sample_index': sample.index, 'teacher': sample.metadata.get('opd_teacher'),
                              'response_tokens': sample.response_length,
                              'total_tokens': len(sample.tokens)}) + '\n')
    if not success:
        _stream.flush()


async def reward_func(args, sample, **kwargs):
    if sample.metadata.get('mopd_evaluation', False):
        return score(sample.response, sample.label)
    sample.metadata['raw_reward'] = score(sample.response, sample.label)
    total_started = time.monotonic()
    for attempt in range(1, ATTEMPTS + 1):
        started = time.monotonic()
        success = False
        error = None
        try:
            result = await teacher_reward(args, sample, **kwargs)
            success = True
            return result
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectError) as exc:
            error = type(exc).__name__
            if attempt == ATTEMPTS:
                raise
            logger.warning('Frozen-teacher scoring retry: sample=%s teacher=%s attempt=%s error=%s',
                           sample.index, sample.metadata.get('opd_teacher'), attempt, error)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            _record_attempt(sample, attempt, started, total_started, success, error)
        await asyncio.sleep(FIRST_RETRY_SECONDS * 2 ** (attempt - 1))


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """Miles' custom rollout log hook: record the training batch's verifier scores and lengths.

    Returns False so Miles still logs its own rollout metrics.
    """
    result = Path(os.environ['CAMPAIGN_RESULT'])
    with (result / 'train-samples.jsonl').open('a') as f:
        for sample in samples:
            f.write(json.dumps({'rollout_id': rollout_id, 'index': sample.index, 'domain': sample.metadata.get('domain'),
                                'teacher': sample.metadata.get('opd_teacher'), 'score': sample.metadata.get('raw_reward'),
                                'prompt_tokens': len(sample.tokens) - sample.response_length,
                                'response_tokens': sample.response_length,
                                'status': getattr(sample.status, 'value', str(sample.status))}) + '\n')
    by_domain, unscored = {}, 0
    for sample in samples:
        value = sample.metadata.get('raw_reward')
        if value is None:  # logging must never stop training
            unscored += 1
            continue
        by_domain.setdefault(sample.metadata.get('domain'), []).append(value)
    record = {'rollout_id': rollout_id, 'update': rollout_id + 1, 'unscored': unscored,
              'domains': {d: {'samples': len(v), 'score': sum(v) / len(v)}
                          for d, v in sorted(by_domain.items(), key=lambda kv: str(kv[0]))}}
    with (result / 'train-reward.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
    return False
