"""Offline checks of the SFT datum, schedules, trace filters, rendering, and benchmark summary; no service calls.

The rendering test needs the pinned dataset snapshot in the Hugging Face cache and is skipped otherwise.
"""
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

os.environ['HF_HUB_OFFLINE'] = '1'  # Before transformers is imported: use only the cached tokenizer.
from sft import (DEFAULTS, DOMAINS, MODEL, RENDERER, REVISION, Run, epoch_order, lr_at, select_traces,  # noqa: E402
                 sequence_nll, sft_datum, summarize)
from shared.puzzles import load_split  # noqa: E402


class SFTTests(unittest.TestCase):
    def test_datum_trains_completion_only(self):
        d = sft_datum([10, 11, 12], [20, 21, 22])
        self.assertEqual(d.model_input.to_ints(), [10, 11, 12, 20, 21])
        self.assertEqual(d.loss_fn_inputs['target_tokens'].data, [11, 12, 20, 21, 22])
        self.assertEqual(d.loss_fn_inputs['weights'].data, [0, 0, 1, 1, 1])

    def test_sequence_nll_from_either_provider(self):
        import tinker
        tinker_output = {'logprobs': tinker.TensorData(data=[-9.0, -9.0, -2.0, -1.0, -0.5], dtype='float32', shape=[5])}
        fireworks_output = {'loss': tinker.TensorData(data=[3.5], dtype='float32', shape=[1])}
        self.assertAlmostEqual(sequence_nll(tinker_output, 3, 3), 3.5)
        self.assertAlmostEqual(sequence_nll(fireworks_output, 3, 3), 3.5)

    def test_schedules(self):
        self.assertEqual(lr_at(0, 10, 1e-3), 1e-3)
        self.assertAlmostEqual(lr_at(5, 10, 1e-3), 5e-4)
        self.assertGreater(lr_at(9, 10, 1e-3), 0)
        self.assertAlmostEqual(lr_at(5, 10, 1e-3, 'cosine'), 5e-4)
        self.assertGreater(lr_at(9, 10, 1e-3, 'cosine'), 0)
        self.assertEqual(lr_at(9, 10, 1e-3, 'constant'), 1e-3)

    def test_first_epoch_is_the_released_teachers_shuffle(self):
        # The released teachers shuffled once with Random(seed + domain index); epoch 0 must stay identical.
        seed = 20260921 + 1
        expected = list(range(50))
        random.Random(seed).shuffle(expected)
        orders = epoch_order(50, 2, seed)
        self.assertEqual(orders[0], expected)
        self.assertNotEqual(orders[1], expected)
        self.assertEqual(sorted(orders[1]), list(range(50)))

    def test_trace_filters(self):
        rows = [{'id': f'{p}-{s}', 'source_index': p, 'sample_index': s, 'correct_of_3': c}
                for p, c in [(0, 3), (1, 1), (2, 2)] for s in range(c)]
        self.assertEqual(select_traces(rows), rows)
        self.assertEqual([r['id'] for r in select_traces(rows, min_correct=2)], ['0-0', '0-1', '0-2', '2-0', '2-1'])
        self.assertEqual([r['id'] for r in select_traces(rows, traces_per_prompt=1)], ['0-0', '1-0', '2-0'])
        subset = select_traces(rows, max_rows=3, seed=1)
        self.assertEqual(len(subset), 3)
        self.assertEqual(subset, select_traces(rows, max_rows=3, seed=1))

    def test_defaults_reproduce_released_teachers(self):
        self.assertEqual((DEFAULTS['revision'], DEFAULTS['epochs'], DEFAULTS['batch_size'], DEFAULTS['rank']),
                         ('b21bb3e75b1ca903f4b69bd609adc89bd99543f1', 1, 128, 64))
        self.assertAlmostEqual(DEFAULTS['learning_rate'], 0.0004990818286656736)

    def test_summary(self):
        records = [{'domain': 'caesar_cipher', 'index': i, 'sample': j, 'score': float(i == 0 or j == 0),
                    'tokens': [0] * 10, 'response': '</think>', 'stop_reason': 'stop'}
                   for i in range(2) for j in range(3)]
        s = summarize(records, 'caesar_cipher', 3)
        self.assertEqual((s['problems'], s['samples'], s['correct']), (2, 6, 4))
        self.assertAlmostEqual(s['accuracy'], 4 / 6)
        self.assertEqual(s['pass_at_3'], 1.0)
        self.assertEqual(s['all_3_correct'], 0.5)

    def test_released_traces_render_to_prompt_plus_completion(self):
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import snapshot_download
            root = snapshot_download(DEFAULTS['dataset'], repo_type='dataset', revision=DEFAULTS['revision'])
        except Exception as exc:
            self.skipTest(f'dataset snapshot not cached: {exc}')
        from transformers import AutoTokenizer
        from tinker_cookbook import renderers
        run = Run.__new__(Run)
        run.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        run.renderer = renderers.get_renderer(RENDERER, run.tokenizer, model_name=MODEL)
        run.args = SimpleNamespace(max_tokens=0)
        end = run.tokenizer.convert_tokens_to_ids('<|im_end|>')
        for domain in DOMAINS:
            path = sorted((Path(root)/'data'/domain).glob('train-*.parquet'))[0]
            rows = pq.read_table(path, columns=['id', 'messages']).slice(0, 4).to_pylist()
            dev_system = load_split(domain, 'dev')[0]['prompt'][0]['content']
            for row in rows:
                prompt, completion = run.tokenize(row)
                self.assertEqual(row['messages'][0]['content'], dev_system)
                self.assertEqual(completion[-1], end)
                text = run.tokenizer.decode(completion)
                self.assertEqual(text.count('</think>'), 1)
                self.assertIn('<answer>', text.rsplit('</think>', 1)[1])


if __name__ == '__main__':
    unittest.main()
