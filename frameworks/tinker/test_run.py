"""Offline checks for the train/eval boundary and token alignment."""
import unittest

from run import centered_advantages, datum, load_data, routed_advantages, scoring
from select_teachers import choose


class TrainingTests(unittest.TestCase):
    def test_teacher_selection_goal_and_fallback(self):
        rows = [{'step':step, 'evaluation':{'countdown':{'accuracy':score}}}
                for step,score in [(10,.3),(20,.5),(30,.6),(40,.5)]]
        self.assertEqual(choose(rows,'countdown',.5)['step'],20)
        selected = choose(rows,'countdown',.7)
        self.assertEqual(selected['step'],30)
        self.assertFalse(selected['target_met'])

    def test_next_token_alignment_and_prompt_mask(self):
        d = datum([10, 11, 12], [20, 21], [-0.5, -0.7], [1.5, -0.2])
        self.assertEqual(d.model_input.to_ints(), [10, 11, 12, 20])
        self.assertEqual(d.loss_fn_inputs['target_tokens'].data, [0, 0, 20, 21])
        for actual, expected in zip(d.loss_fn_inputs['advantages'].data, [0, 0, 1.5, -0.2]):
            self.assertAlmostEqual(actual, expected)

    def test_group_advantages(self):
        self.assertEqual(centered_advantages([0]*8), [0]*8)
        self.assertEqual(centered_advantages([1]*8), [0]*8)
        values = centered_advantages([1, 0, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(sum(values), 0)
        self.assertGreater(values[0], 0)
        self.assertLess(values[1], 0)

    def test_nonfinite_supervision_rejected(self):
        with self.assertRaises(AssertionError):
            datum([10], [20], [-0.5], [float('nan')])

    def test_teacher_alignment_and_reverse_kl_sign(self):
        teacher, adv = routed_advantages(3, [-2.0, -0.1], [None, -5, -8, -1.0, -0.5])
        self.assertEqual(teacher, [-1.0, -0.5])
        self.assertEqual(adv, [1.0, -0.4])
        with self.assertRaises(AssertionError):
            routed_advantages(2, [-1], [None, -5, None])

    def test_exact_data_and_no_dev_training_overlap(self):
        for domain in ('countdown', 'graph_color'):
            train = load_data(domain, 'train')
            dev = load_data(domain, 'dev')
            self.assertEqual(len(train), 10000)
            self.assertEqual(len(dev), 512)
            train_prompts = {str(r['prompt']) for r in train}
            self.assertFalse(train_prompts.intersection(str(r['prompt']) for r in dev))

    def test_countdown_uses_all_numbers(self):
        label = {'domain': 'countdown', 'numbers': [2, 3, 4, 5], 'target': 14}
        self.assertEqual(scoring.score('<answer>2+3+4+5</answer>', label), 1)
        self.assertEqual(scoring.score('<answer>14</answer>', label), 0)


if __name__ == '__main__':
    unittest.main()
