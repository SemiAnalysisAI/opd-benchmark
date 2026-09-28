"""Offline checks of token alignment, data, selection, and pipelining; no service calls.

The scorer test needs reasoning-gym==0.1.25, for example CAMPAIGN_PYDEPS=<its site-packages>.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

os.environ['HF_HUB_OFFLINE'] = '1'  # Before transformers is imported: use only the cached tokenizer.
from run import DOMAINS, MODEL, RENDERER, REVISION, Run, centered_advantages, datum, routed_advantages  # noqa: E402
from shared import recipe, scoring  # run.py puts the repository on sys.path.
from shared.puzzles import load_split  # noqa: E402
from launch import choose, sft_arguments  # noqa: E402


class TrainingTests(unittest.TestCase):
    def test_launcher_settings(self):
        settings = {'learning_rate': 2e-4, 'domains': {'caesar_cipher': {'epochs': 2}}}
        self.assertEqual(sft_arguments(settings), ['--learning-rate', '0.0002'])
        self.assertEqual(sft_arguments(settings, 'caesar_cipher'), ['--learning-rate', '0.0002', '--epochs', '2'])
        self.assertEqual(sft_arguments(settings, 'simple_geometry'), ['--learning-rate', '0.0002'])
        with self.assertRaises(ValueError):
            sft_arguments({'domains': {'countdown': {}}})

    def test_teacher_selection_goal_and_fallback(self):
        rows = [{'step': step, 'evaluation': {'caesar_cipher': {'accuracy': score}}}
                for step, score in [(25, .3), (50, .5), (75, .6), (100, .5)]]
        self.assertEqual(choose(rows, 'caesar_cipher', .5)['step'], 50)
        selected = choose(rows, 'caesar_cipher', .7)
        self.assertEqual(selected['step'], 75)
        self.assertFalse(selected['target_met'])

    def test_next_token_alignment_and_prompt_mask(self):
        d = datum([10, 11, 12], [20, 21], [-0.5, -0.7], [1.5, -0.2])
        self.assertEqual(d.model_input.to_ints(), [10, 11, 12, 20])
        self.assertEqual(d.loss_fn_inputs['target_tokens'].data, [0, 0, 20, 21])
        for actual, expected in zip(d.loss_fn_inputs['advantages'].data, [0, 0, 1.5, -0.2]):
            self.assertAlmostEqual(actual, expected)
        with self.assertRaises(AssertionError):
            datum([10], [20], [-0.5], [float('nan')])

    def test_group_advantages(self):
        self.assertEqual(centered_advantages([0]*8), [0]*8)
        self.assertEqual(centered_advantages([1]*8), [0]*8)
        values = centered_advantages([1, 0, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(sum(values), 0)
        self.assertGreater(values[0], 0)
        self.assertLess(values[1], 0)

    def test_teacher_alignment_and_reverse_kl_sign(self):
        teacher, adv = routed_advantages(3, [-2.0, -0.1], [None, -5, -8, -1.0, -0.5])
        self.assertEqual(teacher, [-1.0, -0.5])
        self.assertEqual(adv, [1.0, -0.4])
        with self.assertRaises(AssertionError):
            routed_advantages(2, [-1], [None, -5, None])

    def test_exact_data_and_no_dev_training_overlap(self):
        for domain in DOMAINS:
            train, dev = load_split(domain, 'train'), load_split(domain, 'dev')
            self.assertEqual((len(train), len(dev)), (recipe.TRAIN_EXAMPLES_PER_DOMAIN, recipe.EVAL_EXAMPLES[domain]))
            self.assertFalse({str(r['prompt']) for r in train}.intersection(str(r['prompt']) for r in dev))

    def test_thinking_prompt_matches_pinned_template(self):
        from transformers import AutoTokenizer
        from tinker_cookbook import renderers
        r = Run.__new__(Run)
        r.args = SimpleNamespace(max_tokens=recipe.MAX_RESPONSE_TOKENS)
        r.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        r.renderer = renderers.get_renderer(RENDERER, r.tokenizer, model_name=MODEL)
        for domain in DOMAINS:
            ids = r.prompt(load_split(domain, 'dev')[0])  # Asserts HF == renderer tokens and the length bound.
            self.assertTrue(r.tokenizer.decode(ids).endswith('<|im_start|>assistant\n<think>\n'))

    def test_scoring_needs_finished_thinking(self):
        row = load_split('caesar_cipher', 'dev')[0]
        answer = f"<answer>{json.loads(row['label'])['answer']}</answer>"
        self.assertEqual(scoring.score(f'plan</think>\n\n{answer}', row['label']), 1)
        self.assertEqual(scoring.score(f'still thinking {answer}', row['label']), 0)  # Cut off mid-thought.
        self.assertEqual(scoring.score('plan</think>\n\n<answer>WRONG</answer>', row['label']), 0)


def teacher_client(callback=lambda: None):
    def compute(model_input):
        callback()
        return SimpleNamespace(result=lambda: [None, -3.0, -1.0])
    return SimpleNamespace(compute_logprobs=compute)


class PipelineTests(unittest.TestCase):
    def make_run(self, directory, prompts=4):
        r = Run.__new__(Run)
        r.out = Path(directory)
        r.started = time.monotonic()
        r.write_lock = threading.Lock()
        r.pool, r.teacher_pool = ThreadPoolExecutor(max_workers=2), ThreadPoolExecutor(max_workers=2)
        r.prepare_pool = ThreadPoolExecutor(max_workers=1)
        for pool in (r.pool, r.teacher_pool, r.prepare_pool):
            self.addCleanup(pool.shutdown)
        r.args = SimpleNamespace(prompts=prompts, seed=7, mode='mopd', checkpoint=None,
            rank=64, domain=None, start_step=0, steps=3, pipeline_depth=1,
            group_size=1, skip_baseline=False, learning_rate=1e-5,
            save_every=2, eval_every=2, target_score=None)
        r.rows = {d: {'train': [{'_index': i} for i in range(6)]} for d in DOMAINS}
        r.order = {d: list(range(6)) for d in DOMAINS}
        r.sample_one = lambda client, domain, row, group, temperature, seed: [{'domain': domain,
            'index': row['_index'], 'sample': 0, 'prompt_tokens': [10, 11], 'tokens': [20], 'logprobs': [-2.0],
            'response': 'x', 'score': 0, 'stop_reason': 'stop', 'request_seconds': 0.0}]
        return r

    def test_teacher_starts_before_all_generation_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            r = self.make_run(directory, 2)
            teacher_started = threading.Event()
            original = r.sample_one

            def sample(client, domain, *args):
                if domain == DOMAINS[1]:
                    self.assertTrue(teacher_started.wait(3), 'teacher scoring did not overlap generation')
                return original(client, domain, *args)
            r.sample_one = sample
            groups, info = r.prepare_mopd(None, {d: teacher_client(teacher_started.set) for d in DOMAINS}, 0, 0)
            self.assertEqual([g[0]['domain'] for g in groups], list(DOMAINS))
            self.assertTrue(all(g[0]['advantages'] == [1.0] for g in groups))
            self.assertEqual(info['policy_version'], 0)

    def test_lookahead_bound_versions_budget_and_request_order(self):
        with tempfile.TemporaryDirectory() as directory:
            r = self.make_run(directory)
            r.args.teachers = r.out/'routes.json'
            r.args.teachers.write_text(json.dumps({d: 'teacher/'+d for d in DOMAINS}))
            updates, submitted = [], []
            prepared = {i: threading.Event() for i in range(3)}
            prepare = r.prepare_mopd

            def wrapped(client, teachers, step, version):
                prepared[step].set()
                return prepare(client, teachers, step, version)
            r.prepare_mopd = wrapped

            class Train:
                version = 0

                def get_info(self):
                    return SimpleNamespace(model_dump_json=lambda **kw: '{"model_id":"student"}')

                def save_weights_and_get_sampling_client(self):
                    return self.version

                def forward_backward(self, datums, **kw):
                    current = self.version
                    submitted.append(('fb', current))

                    def result():
                        assert submitted[-1] == ('opt', current)
                        if current < 2:
                            assert prepared[current+1].wait(3), 'next batch did not start before learner wait'
                        return SimpleNamespace(loss_fn_outputs=[{'logprobs': SimpleNamespace(data=[0.0, -2.0])}
                                                                for _ in datums], metrics={})
                    return SimpleNamespace(result=result)

                def optim_step(self, params):
                    submitted.append(('opt', self.version))

                    def result():
                        self.version += 1
                        updates.append(self.version)
                        return SimpleNamespace(metrics={})
                    return SimpleNamespace(result=result)
            train = Train()
            r.service = SimpleNamespace(create_lora_training_client=lambda *a, **kw: train,
                                        create_sampling_client=lambda **kw: teacher_client())
            evaluations, checkpoints = [], []
            r.evaluate = lambda client, step: evaluations.append((client, step))
            r.checkpoint = lambda client, step: checkpoints.append(step)
            r.execute()
            metrics = [json.loads(x) for x in (r.out/'metrics.jsonl').read_text().splitlines()]
            samples = [json.loads(x) for x in (r.out/'samples.jsonl').read_text().splitlines()]
            self.assertEqual(updates, [1, 2, 3])
            self.assertEqual([m['accepted_policy_lag'] for m in metrics], [0, 1, 1])
            self.assertEqual([m['pipeline']['policy_version'] for m in metrics], [0, 0, 1])
            self.assertEqual(len(samples), 12)
            for step in range(3):
                for domain in DOMAINS:
                    self.assertEqual(sum(x['step'] == step and x['domain'] == domain for x in samples), 2)
            self.assertEqual(evaluations, [(0, 0), (2, 2), (3, 3)])
            self.assertEqual(checkpoints, [2, 3])


if __name__ == '__main__':
    unittest.main()
