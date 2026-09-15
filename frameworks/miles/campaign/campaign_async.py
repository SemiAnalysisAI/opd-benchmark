"""Record scoring attempts and retry transient reads from frozen teachers."""
import asyncio
import json
import logging
import os
from pathlib import Path
import time

import httpx

from examples.mopd_puzzles.tasks import reward_func as verifier_reward
from miles.rollout.on_policy_distillation import reward_func as teacher_reward

_stream = None
logger = logging.getLogger(__name__)


def _record_attempt(sample, attempt, started, total_started, success, error):
    global _stream
    if _stream is None:
        path = Path(os.environ['MOPD_RESULT']) / f'teacher-latency-{os.getpid()}.jsonl'
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
        return await verifier_reward(args, sample, **kwargs)
    total_started = time.monotonic()
    for attempt in range(1, 4):
        started = time.monotonic()
        success = False
        error = None
        try:
            result = await teacher_reward(args, sample, **kwargs)
            success = True
            return result
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectError) as exc:
            error = type(exc).__name__
            if attempt == 3:
                raise
            logger.warning('Frozen-teacher scoring retry: sample=%s teacher=%s attempt=%s error=%s',
                           sample.index, sample.metadata.get('opd_teacher'), attempt, error)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            _record_attempt(sample, attempt, started, total_started, success, error)
        await asyncio.sleep(0.25 * 2 ** (attempt - 1))
