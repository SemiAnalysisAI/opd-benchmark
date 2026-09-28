"""Equal-domain prompts and batches for Miles, for any `recipe.DOMAINS`.

BalancedDataSource interleaves the domains of the mixed training file (replacing the pinned
launcher's puzzle-specific source). BalancedAsyncBuffer assembles equal-domain async batches
without bypassing the OPD or policy-lag guards.
"""

import asyncio
from collections import Counter, defaultdict
import time

from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.rollout.filter_hub.base_types import call_dynamic_filter
from miles.rollout.fully_async_data_buffer import DefaultDataBuffer, first_sample, iter_samples
from miles.utils.types import Sample
from shared import recipe

DOMAIN_WAIT_SECONDS = 3600.0  # A 30k-token thinking rollout can take many minutes.


class _InterleavedDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self._interleave()

    def _interleave(self):
        groups = defaultdict(list)
        for sample in self.dataset.samples:
            groups[sample.metadata['domain']].append(sample)
        if set(groups) != set(recipe.DOMAINS) or len({len(g) for g in groups.values()}) != 1:
            raise ValueError(f'Expected equally sized {recipe.DOMAINS} datasets, got {sorted(groups)}')
        self.samples = [s for group in zip(*(groups[d] for d in recipe.DOMAINS), strict=True) for s in group]

    def shuffle(self, epoch_id):
        self.dataset.shuffle(epoch_id)
        self._interleave()

    def __len__(self):
        return len(self.samples)


class BalancedDataSource(RolloutDataSourceWithBuffer):
    def __init__(self, args):
        super().__init__(args)
        if self.dataset is None or args.n_samples_per_prompt != 1:
            raise ValueError('Balanced OPD requires a global dataset and one response per prompt')
        self.dataset = _InterleavedDataset(self.dataset)


class BalancedAsyncBuffer(DefaultDataBuffer):
    """Reserve an equal share of the bounded queue for each puzzle domain."""

    def __init__(self, input):
        super().__init__(input)
        args = input.args
        self._domains = recipe.DOMAINS
        count = len(self._domains)
        if args.n_samples_per_prompt != 1 or args.rollout_batch_size % count:
            raise ValueError("Balanced async MOPD requires one sample per prompt and a batch divisible by the domains")
        if args.rollout_sample_filter_path:
            raise ValueError("A filter after batch assembly could invalidate domain balance")
        if self._capacity < count or self._capacity % count:
            raise ValueError("Balanced async MOPD requires a buffer capacity divisible by the domains")
        self._per_domain_capacity = self._capacity // count
        self._next_domain_index = 0
        self._window = Counter()
        self._wait_timeout = DOMAIN_WAIT_SECONDS

    def _domain(self, entry):
        if len(entry.group) != 1 or isinstance(entry.group[0], list):
            raise ValueError("Balanced async MOPD requires a single flat sample per group")
        domain = first_sample(entry.group).metadata.get("domain")
        if domain not in self._domains:
            raise ValueError(f"Unexpected OPD domain {domain!r}")
        return domain

    @staticmethod
    def _active_tokens(entry):
        sample = first_sample(entry.group)
        if sample.remove_sample:
            return 0
        if sample.loss_mask is None:
            return sample.response_length
        if len(sample.loss_mask) != sample.response_length:
            raise ValueError("The response and loss mask lengths differ")
        return sum(sample.loss_mask)

    async def put(self, input):
        domain = self._domain(input)
        if any(s.status == Sample.Status.ABORTED for s in iter_samples(input.group)):
            self._metric_aborted_groups += 1
            self._unused_handler_fn(input.prompt_group)
            return
        if self._active_tokens(input) <= 0:
            self._window[f"{domain}/inactive_groups_filtered"] += 1
            self._unused_handler_fn(input.prompt_group)
            return
        verdict = call_dynamic_filter(self._dynamic_filter, self._args, input.group)
        if not verdict.keep:
            self._metric_gatherer.on_dynamic_filter_drop(reason=verdict.reason)
            return

        async with self._cond:
            # A full queue contains both domains, so the consumer can make progress.
            while len(self._buffer) >= self._capacity:
                await self._cond.wait()
            positions = [i for i, entry in enumerate(self._buffer) if self._domain(entry) == domain]
            if len(positions) >= self._per_domain_capacity:
                # Blocking here would stop the serial producer from adding the missing domain.
                evicted = self._buffer.pop(positions[0])
                self._window[f"{domain}/overflow_groups_evicted"] += 1
                self._unused_handler_fn(evicted.prompt_group)
            self._buffer.append(input)
            self._cond.notify_all()

    async def get(self, current_version=None, **_):
        if current_version is None:
            raise ValueError("Balanced async MOPD requires the current policy version")
        self._current_version = current_version
        domain = self._domains[self._next_domain_index]
        started = time.monotonic()
        async with self._cond:
            while True:
                position = next((i for i, e in enumerate(self._buffer) if self._domain(e) == domain), None)
                if position is None:
                    remaining = self._wait_timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError(f"No eligible {domain} sample arrived within {self._wait_timeout}s")
                    try:
                        await asyncio.wait_for(self._cond.wait(), remaining)
                    except asyncio.TimeoutError as exc:
                        raise TimeoutError(f"No eligible {domain} sample arrived within {self._wait_timeout}s") from exc
                    continue
                entry = self._buffer.pop(position)
                self._cond.notify_all()
                lag = self._staleness(entry.group, current_version)
                if lag is None or lag < 0:
                    raise ValueError(f"The {domain} sample has an absent or future policy version")
                if self._args.max_weight_staleness is not None and lag > self._args.max_weight_staleness:
                    self._metric_stale_groups += 1
                    self._window[f"{domain}/stale_groups_filtered"] += 1
                    self._unused_handler_fn(entry.prompt_group)
                    continue
                active = self._active_tokens(entry)
                if active <= 0:
                    self._window[f"{domain}/inactive_groups_filtered"] += 1
                    self._unused_handler_fn(entry.prompt_group)
                    continue
                self._metric_consumed_staleness.append(lag)
                self._window[f"{domain}/accepted_groups"] += 1
                self._window[f"{domain}/accepted_active_tokens"] += active
                self._window[f"{domain}/wait_seconds"] += time.monotonic() - started
                self._next_domain_index = (self._next_domain_index + 1) % len(self._domains)
                return entry

    def get_metrics(self):
        metrics = super().get_metrics()
        prefix = "rollout/fully_async/domain_balance/"
        for domain in self._domains:
            metrics[f"{prefix}{domain}/queued_groups"] = sum(self._domain(e) == domain for e in self._buffer)
            for key in ("accepted_groups", "accepted_active_tokens", "wait_seconds", "inactive_groups_filtered",
                        "stale_groups_filtered", "overflow_groups_evicted"):
                metrics[f"{prefix}{domain}/{key}"] = self._window[f"{domain}/{key}"]
        self._window.clear()
        return metrics
