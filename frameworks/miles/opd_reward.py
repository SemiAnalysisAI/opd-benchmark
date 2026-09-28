"""Miles reward hook: score with the routed frozen teacher, retrying transient transport errors.

Evaluation samples use the shared task verifier instead. Training samples also get the verifier's
score as `metadata["raw_reward"]`, which Miles logs per step as `rollout/raw_reward` in place of the
teacher reward's constant 0; the OPD loss reads the teacher reward, not this. Every teacher attempt is
recorded in `teacher-latency-<pid>.jsonl`.
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
