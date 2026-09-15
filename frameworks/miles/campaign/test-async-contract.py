"""Execute the changed pure functions without importing the GPU runtime."""
import ast
import dataclasses
import shlex
import types
import typing
import unittest
from pathlib import Path

root = Path(__file__).parent / 'source'


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if getattr(node, 'name', None) in names]
    assert len(selected) == len(names)
    tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected]
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return namespace


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.functions = extract(root / 'miles/rollout/generate_utils/generate_endpoint_utils.py',
                                 {'_student_opd_top_k', '_retain_student_opd_scores'}, {})
        self.args = types.SimpleNamespace(use_opd=True, opd_top_k_strategy='only-student', opd_log_prob_top_k=2)

    def test_topk_request_selection(self):
        fn = self.functions['_student_opd_top_k']
        self.assertEqual(fn(self.args), 2)
        self.args.use_opd = False
        self.assertEqual(fn(self.args), 0)
        self.args.use_opd = True
        self.args.opd_top_k_strategy = 'only-teacher'
        self.assertEqual(fn(self.args), 0)

    def test_scores_append_without_replacing_old_behavior_scores(self):
        fn = self.functions['_retain_student_opd_scores']
        sample = types.SimpleNamespace(metadata={'domain': 'countdown'})
        rows = [[[-0.3, 12], [-1.4, 13]]]
        output = {'meta_info': {'output_token_logprobs': [[-0.3, 12]], 'output_top_logprobs': rows}}
        fn(self.args, sample, output)
        fn(self.args, sample, output)
        self.assertEqual(sample.metadata['opd_student_top_logprobs'], rows + rows)
        self.assertEqual(sample.metadata['domain'], 'countdown')

    def test_missing_and_misaligned_scores_fail(self):
        fn = self.functions['_retain_student_opd_scores']
        for rows in (None, [], [[[-0.3, 12]]]):
            sample = types.SimpleNamespace(metadata={})
            output = {'meta_info': {'output_token_logprobs': [[-0.3, 12]], 'output_top_logprobs': rows}}
            with self.assertRaises(ValueError):
                fn(self.args, sample, output)

    def test_empty_aborted_response_is_valid(self):
        sample = types.SimpleNamespace(metadata={})
        self.functions['_retain_student_opd_scores'](self.args, sample, {'meta_info': {}})
        self.assertEqual(sample.metadata, {})

    def test_async_launcher_contract(self):
        namespace = {'dataclass': dataclasses.dataclass, 'Literal': typing.Literal, 'shlex': shlex,
                     'U': types.SimpleNamespace(ExecuteTrainConfig=object, create_run_id=lambda: 'test')}
        extract(root / 'scripts/run_mopd_puzzles.py', {'ScriptArgs', '_training_args'}, namespace)
        cls = namespace['ScriptArgs']
        with self.assertRaises(ValueError):
            cls(fully_async=True)
        args = cls(mode='student', fully_async=True, colocate=False, use_rollout_logprobs=True,
                   rollout_gpus=6, rollout_gpus_per_engine=2, actor_gpus=8, num_rollout=40)
        argv = shlex.split(namespace['_training_args'](args))
        self.assertIn('--fully-async', argv)
        self.assertIn('--use-rollout-logprobs', argv)
        self.assertNotIn('--colocate', argv)
        self.assertNotIn('--rollout-function-path', argv)
        self.assertNotIn('--eval-function-path', argv)
        for flag, value in {'--rollout-num-gpus': '6', '--rollout-num-gpus-per-engine': '2',
                            '--sglang-ep-size': '2', '--rollout-batch-size': '128',
                            '--global-batch-size': '128', '--n-samples-per-prompt': '1',
                            '--opd-log-prob-top-k': '16', '--num-rollout': '40'}.items():
            self.assertEqual(argv[argv.index(flag) + 1], value)

    def test_endpoint_calls_retention_and_requests_topk(self):
        tree = ast.parse((root / 'miles/rollout/generate_utils/generate_endpoint_utils.py').read_text())
        functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertIn('_retain_student_opd_scores', ast.unparse(functions['update_sample_from_response']))
        self.assertIn('top_logprobs_num', ast.unparse(functions['compute_request_payload']))


if __name__ == '__main__':
    unittest.main()
