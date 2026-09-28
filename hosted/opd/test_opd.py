"""Offline checks of the distillation math and prompt batches; no service calls."""
import os
from types import SimpleNamespace
import unittest

os.environ['HF_HUB_OFFLINE'] = '1'
from common import Experiment, advantages, percentile, policy_datum  # noqa: E402
from shared import recipe  # noqa: E402
from shared.puzzles import load_split  # noqa: E402


class DistillationTests(unittest.TestCase):
    def test_advantage_is_negative_reverse_kl(self):
        # The teacher likes token 0 more than the student does (positive) and token 1 less (negative).
        self.assertEqual(advantages([-1.0, -3.0], [-2.0, -1.0]), [1.0, -2.0])
        with self.assertRaises(AssertionError):
            advantages([-1.0], [-1.0, -2.0])
        with self.assertRaises(AssertionError):
            advantages([float('nan')], [-1.0])

    def test_datum_alignment(self):
        d = policy_datum([10, 11, 12], [20, 21], [-0.5, -0.7], [1.5, -0.2])
        self.assertEqual(d.model_input.to_ints(), [10, 11, 12, 20])
        self.assertEqual(d.loss_fn_inputs['target_tokens'].data, [0, 0, 20, 21])
        self.assertEqual(d.loss_fn_inputs['advantages'].data[:2], [0.0, 0.0])

    def test_opd_and_mopd_see_one_caesar_prompt_order(self):
        import random
        run = Experiment.__new__(Experiment)
        run.args = SimpleNamespace(seed=20260921)
        run.data = {d: {'train': load_split(d, 'train')} for d in recipe.DOMAINS}
        run.order = {}
        for d in recipe.DOMAINS:
            run.order[d] = list(range(len(run.data[d]['train'])))
            random.Random(run.args.seed + recipe.DOMAINS.index(d)).shuffle(run.order[d])
        # OPD takes 128 caesar prompts per update; MOPD takes 64 per domain. Caesar's order is the same.
        opd = [r['_index'] for _, r in run.train_prompts('caesar_cipher', 0, 128)]
        mopd = [r['_index'] for u in (0, 1) for _, r in run.train_prompts('caesar_cipher', u, 64)]
        self.assertEqual(opd, mopd)
        self.assertEqual(len(set(opd)), 128)

    def test_policy_lag(self):
        """Async samples update u+1 on update u's sampler (lag 1); synchronous samples it on its own (lag 0)."""
        from concurrent.futures import ThreadPoolExecutor
        from itertools import count

        for lag, expected in ((1, ['w0', 'w0', 'w1', 'w2']), (0, ['w0', 'w1', 'w2', 'w3'])):
            done = SimpleNamespace(result=lambda: SimpleNamespace(loss_fn_outputs=[{}], metrics={}))
            train = SimpleNamespace(forward_backward=lambda *a, **k: done, optim_step=lambda *a, **k: done)
            version = count(1)
            run = Experiment.__new__(Experiment)
            run.args = SimpleNamespace(policy_lag=lag, updates=4, start_update=0, seed=0, learning_rate=1e-5,
                                       save_every=100, eval_every=100, no_eval=True, skip_initial_eval=True)
            run.pool = ThreadPoolExecutor(4)
            run.backend = SimpleNamespace(sync=lambda train, name: (f'w{next(version)}', name))
            run.start = lambda: (train, 'w0')
            run.rollout = lambda student, *rest: {'student': student, 'prompt_tokens': [1], 'tokens': [2],
                                                  'logprobs': [-1.0], 'advantages': [0.0]}
            run.log = run.save_rollouts = run.append = run.checkpoint = lambda *a, **k: None
            metrics = []
            run.update_metrics = lambda update, rollouts, *rest: metrics.append((rollouts[0]['student'], rest[-1]))
            run.train_loop(lambda update: [('caesar_cipher', {})])
            self.assertEqual(metrics, [(w, 0 if i == 0 else lag) for i, w in enumerate(expected)])

    def test_percentile(self):
        self.assertEqual(percentile(list(range(10)), 0.9), 9)
        self.assertEqual(percentile([5], 0.5), 5)


if __name__ == '__main__':
    unittest.main()
