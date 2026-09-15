"""Exercise the native async rollout consumer and the actual OPD balancing guard."""

import asyncio
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent / "source")]

from miles.rollout.base_types import RolloutFnTrainInput
from miles.rollout.opd_balance import set_domain_weights
from tests.fast.rollout.test_fully_async_rollout import FakeDataSource, make_args, make_fn, make_group


async def test_native_rollout_preserves_active_domains(monkeypatch):
    class Source(FakeDataSource):
        def get_samples(self, num_samples):
            self.next_group_index += 1
            sample = make_group(self.next_group_index, weight_versions=["1"])[0]
            sample.metadata = {"domain": "countdown" if self.next_group_index % 2 else "graph_color"}
            return [[sample]]

    async def generate(state, group, **kwargs):
        await asyncio.sleep(0.001 if group[0].metadata["domain"] == "countdown" else 0.01)
        return group

    args = make_args(rollout_batch_size=128, n_samples_per_prompt=1, async_max_concurrent_samples=32,
                     async_data_buffer_capacity_factor=1, max_weight_staleness=2,
                     custom_async_data_buffer_path="balanced_async_buffer.BalancedAsyncBuffer",
                     opd_domain_balance="static", opd_domain_targets=["countdown=0.5", "graph_color=0.5"],
                     calculate_per_token_loss=False)
    fn = make_fn(monkeypatch, args, Source(), generate=generate)
    try:
        for step in range(3):
            output = await asyncio.wait_for(fn(RolloutFnTrainInput(rollout_id=step, weight_version=1)), 10)
            samples = [group[0] for group in output.samples]
            set_domain_weights(args, samples)
            assert len(samples) == 128
            assert sum(s.metadata["domain"] == "countdown" for s in samples) == 64
            assert all(s.opd_loss_weights.sum().item() > 0 for s in samples)
            assert output.metrics["rollout/fully_async/domain_balance/graph_color/accepted_groups"] == 64
    finally:
        if fn._worker is not None:
            fn._worker.cancel()
            await asyncio.gather(fn._worker, return_exceptions=True)
