"""Execute the real campaign adapter with controlled teacher responses."""
import ast
import asyncio
import io
import json
import logging
from pathlib import Path
import time
import types
import unittest


class ReadError(Exception):
    pass


class RemoteProtocolError(Exception):
    pass


class ConnectError(Exception):
    pass


class RetryTests(unittest.TestCase):
    def run_adapter(self, responses, evaluation=False):
        calls = []
        sleeps = []
        stream = io.StringIO()

        async def teacher(args, sample, **kwargs):
            calls.append((tuple(sample.tokens), sample.metadata.copy()))
            value = responses[len(calls) - 1]
            if isinstance(value, BaseException):
                raise value
            return value

        async def verifier(*args, **kwargs):
            return 'verifier'

        async def sleep(seconds):
            sleeps.append(seconds)

        namespace = {'teacher_reward': teacher, 'verifier_reward': verifier, '_stream': stream,
                     'json': json, 'time': time, 'logger': logging.getLogger('test'),
                     'httpx': types.SimpleNamespace(ReadError=ReadError, RemoteProtocolError=RemoteProtocolError,
                                                    ConnectError=ConnectError),
                     'asyncio': types.SimpleNamespace(sleep=sleep)}
        tree = ast.parse(Path(__file__).with_name('campaign_async.py').read_text())
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        exec(compile(tree, 'campaign_async.py', 'exec'), namespace)
        sample = types.SimpleNamespace(tokens=[1, 2, 3], response_length=1, index=11,
                                       metadata={'opd_teacher': 'graph_color', 'mopd_evaluation': evaluation})
        try:
            result = asyncio.run(namespace['reward_func'](None, sample))
        except BaseException as exc:
            result = exc
        return result, calls, sleeps, [json.loads(line) for line in stream.getvalue().splitlines()]

    def test_success_has_no_retry(self):
        result, calls, sleeps, rows = self.run_adapter([{'teacher': 'scores'}])
        self.assertEqual(result, {'teacher': 'scores'})
        self.assertEqual(len(calls), 1)
        self.assertEqual(sleeps, [])
        self.assertTrue(rows[0]['success'])

    def test_read_error_retries_same_tokens_and_teacher(self):
        result, calls, sleeps, rows = self.run_adapter([ReadError(), RemoteProtocolError(), 'scores'])
        self.assertEqual(result, 'scores')
        self.assertEqual(calls, [calls[0]] * 3)
        self.assertEqual(sleeps, [0.25, 0.5])
        self.assertEqual([row['success'] for row in rows], [False, False, True])
        self.assertEqual([row['attempt'] for row in rows], [1, 2, 3])

    def test_exhausted_transport_errors_propagate(self):
        result, calls, sleeps, rows = self.run_adapter([ConnectError()] * 3)
        self.assertIsInstance(result, ConnectError)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(rows), 3)

    def test_invalid_scores_and_cancellation_are_not_retried(self):
        for error in (ValueError('bad scores'), asyncio.CancelledError(), asyncio.TimeoutError()):
            result, calls, sleeps, rows = self.run_adapter([error])
            self.assertIsInstance(result, type(error))
            self.assertEqual(len(calls), 1)
            self.assertEqual(sleeps, [])
            self.assertEqual(rows[0]['error_type'], type(error).__name__)

    def test_evaluation_uses_verifier_without_teacher(self):
        result, calls, sleeps, rows = self.run_adapter([], evaluation=True)
        self.assertEqual(result, 'verifier')
        self.assertEqual(calls, [])
        self.assertEqual(rows, [])


if __name__ == '__main__':
    unittest.main()
