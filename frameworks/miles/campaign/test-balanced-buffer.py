"""Exercise the real buffer with controlled completion order and policy versions."""

import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "source")]
if importlib.util.find_spec("torch") is None:
    # Only the heavy Sample type is stubbed on the laptop; buffer code stays real.
    types_stub = ModuleType("miles.utils.types")
    types_stub.Sample = type("Sample", (), {"Status": SimpleNamespace(ABORTED="aborted", COMPLETED="completed")})
    sys.modules["miles.utils.types"] = types_stub

from balanced_async_buffer import BalancedAsyncBuffer
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample


def make_buffer(**overrides):
    args = SimpleNamespace(async_data_buffer_capacity_factor=1, rollout_batch_size=4,
                           n_samples_per_prompt=1, dynamic_sampling_filter_path=None,
                           rollout_sample_filter_path=None, max_weight_staleness=2)
    args.__dict__.update(overrides)
    unused = []
    return BalancedAsyncBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=unused.append)), unused


def entry(domain, index=0, version=1, **kwargs):
    sample = SimpleNamespace(metadata={"domain": domain}, index=index, oldest_weight_version=version,
                             response_length=2, loss_mask=None, remove_sample=False,
                             status=Sample.Status.COMPLETED)
    sample.__dict__.update(kwargs)
    return DataBufferInput(prompt_group=[sample], group=[sample])


class BufferTests(unittest.IsolatedAsyncioTestCase):
    async def test_imbalanced_completion_produces_balanced_batches(self):
        buffer, _ = make_buffer(rollout_batch_size=128)

        async def produce():
            for round_id in range(2):
                for domain in ("countdown", "graph_color"):
                    for i in range(64):
                        await buffer.put(entry(domain, round_id * 1000 + i))
                        await asyncio.sleep(0)

        async def consume():
            for _ in range(2):
                rows = [await buffer.get(current_version=1) for _ in range(128)]
                self.assertEqual(sum(r.group[0].metadata["domain"] == "countdown" for r in rows), 64)
                metrics = buffer.get_metrics()
                self.assertEqual(metrics["rollout/fully_async/domain_balance/graph_color/accepted_groups"], 64)

        await asyncio.wait_for(asyncio.gather(produce(), consume()), 3)

    async def test_missing_domain_does_not_block_producer(self):
        buffer, unused = make_buffer()
        for i in range(20):
            await asyncio.wait_for(buffer.put(entry("countdown", i)), 0.2)
        self.assertEqual(len(buffer._buffer), 2)
        self.assertEqual(len(unused), 18)
        await buffer.put(entry("graph_color"))
        self.assertEqual((await buffer.get(current_version=1)).group[0].metadata["domain"], "countdown")
        self.assertEqual((await buffer.get(current_version=1)).group[0].metadata["domain"], "graph_color")

    async def test_full_buffer_backpressure_resumes(self):
        buffer, _ = make_buffer()
        for domain in ("countdown", "graph_color"):
            for _ in range(2):
                await buffer.put(entry(domain))
        put = asyncio.create_task(buffer.put(entry("graph_color")))
        await asyncio.sleep(0)
        self.assertFalse(put.done())
        await buffer.get(current_version=1)
        await asyncio.wait_for(put, 0.2)
        self.assertLessEqual(len(buffer._buffer), 4)

    async def test_stale_domain_is_replenished_and_accepted_lag_is_clean(self):
        buffer, unused = make_buffer()
        await buffer.put(entry("countdown", version=1))
        get = asyncio.create_task(buffer.get(current_version=4))
        await asyncio.sleep(0.01)
        self.assertFalse(get.done())
        await buffer.put(entry("countdown", version=2))
        self.assertEqual((await get).group[0].oldest_weight_version, 2)
        self.assertEqual(len(unused), 1)
        metrics = buffer.get_metrics()
        self.assertEqual(metrics["rollout/fully_async/max_staleness"], 2)
        self.assertEqual(metrics["rollout/fully_async/stale_groups_filtered"], 1)

    async def test_inactive_removed_and_aborted_samples_are_excluded(self):
        buffer, unused = make_buffer()
        for kwargs in ({"loss_mask": [0, 0]}, {"remove_sample": True},
                       {"response_length": 0}, {"status": Sample.Status.ABORTED}):
            await buffer.put(entry("countdown", **kwargs))
        self.assertEqual(len(unused), 4)
        self.assertFalse(buffer._buffer)
        await buffer.put(entry("countdown", loss_mask=[0, 1]))
        await buffer.get(current_version=1)
        metrics = buffer.get_metrics()
        self.assertEqual(metrics["rollout/fully_async/domain_balance/countdown/accepted_active_tokens"], 1)

    async def test_domain_wait_has_bounded_failure(self):
        buffer, _ = make_buffer()
        buffer._wait_timeout = 0.02
        with self.assertRaisesRegex(TimeoutError, "countdown"):
            await buffer.get(current_version=1)

    async def test_absent_and_future_versions_fail(self):
        for version in (None, 3):
            buffer, _ = make_buffer()
            await buffer.put(entry("countdown", version=version))
            with self.assertRaises(ValueError):
                await buffer.get(current_version=1)

    async def test_metrics_reset_and_configuration_guards(self):
        buffer, _ = make_buffer()
        await buffer.put(entry("countdown"))
        await buffer.get(current_version=1)
        self.assertEqual(buffer.get_metrics()["rollout/fully_async/domain_balance/countdown/accepted_groups"], 1)
        self.assertEqual(buffer.get_metrics()["rollout/fully_async/domain_balance/countdown/accepted_groups"], 0)
        for kwargs in ({"n_samples_per_prompt": 2}, {"rollout_batch_size": 3},
                       {"rollout_sample_filter_path": "filter"}, {"async_data_buffer_capacity_factor": 0.75}):
            with self.assertRaises(ValueError):
                make_buffer(**kwargs)
        with self.assertRaises(ValueError):
            await buffer.put(entry("unknown"))


if __name__ == "__main__":
    unittest.main()
