import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import shlex
import subprocess
import sys
import unittest
import unittest.mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare', ROOT/'tools/prepare.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PackageTests(unittest.TestCase):
    def test_campaign_files_compile_without_machine_values(self):
        for framework in prepare.FRAMEWORKS:
            for source, target in prepare.campaign_files(framework):
                text = source.read_text()
                self.assertNotIn('/Users/', text, str(source))
                self.assertNotIn('clustermax-campaigns', text, str(source))
                if source.suffix == '.py':
                    ast.parse(text, filename=str(source))
            targets = {str(t) for _, t in prepare.campaign_files(framework)}
            self.assertIn('node.py', targets)
            self.assertIn('shared/slurm.py', targets)
            self.assertFalse(any(t.startswith('upstream') for t in targets))

    def test_dataset_integrity_and_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)/'data'
            prepare.materialize_data(dest)
            manifest = prepare.puzzles.manifest()
            for name, entry in manifest.items():
                rows = [json.loads(line) for line in (dest/name).read_text().splitlines()]
                self.assertEqual(len(rows), entry['rows'])
            self.assertEqual(manifest['mixed-train.jsonl']['rows'], 2 * prepare.recipe.TRAIN_EXAMPLES_PER_DOMAIN)
            recipe = prepare.recipe
            for domain in recipe.DOMAINS:
                self.assertEqual(manifest[recipe.data_file(domain, 'dev')]['rows'], recipe.EVAL_EXAMPLES[domain])
                self.assertEqual(manifest[recipe.data_file(domain, 'train')]['rows'], recipe.TRAIN_EXAMPLES_PER_DOMAIN)
                rows = [json.loads(line) for line in (dest/recipe.data_file(domain, 'dev')).read_text().splitlines()]
                self.assertTrue(all(json.loads(r['label'])['task'] == r['metadata']['domain'] == domain for r in rows))

    def test_patch_integrity(self):
        for name in prepare.FRAMEWORKS:
            root = ROOT/'frameworks'/name/'upstream'
            manifest = json.loads((root/'manifest.json').read_text())
            if (root/'source.patch').exists() or manifest['source_patch_sha256'] is not None:
                self.assertEqual(hashlib.sha256((root/'source.patch').read_bytes()).hexdigest(), manifest['source_patch_sha256'])
            self.assertEqual(len(manifest['revision']), 40)

    def test_invalid_site_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad.json'
            path.write_text('{}')
            with self.assertRaises(ValueError):
                prepare.load_site(path)
            site = json.loads((ROOT/'config/site.example.json').read_text())
            site['base_model'] = '/tmp/model\";print(1)'
            path.write_text(json.dumps(site))
            with self.assertRaises(ValueError):
                prepare.load_site(path)

    def test_existing_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                prepare.prepare('slime', prepare.load_site(ROOT/'config/site.example.json'), Path(temp))

    def test_submit_defaults_to_printing_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'site.json').write_text((ROOT/'config/site.example.json').read_text())
            (root/'package-provenance.json').write_text('{"framework":"slime"}')
            result = subprocess.run([sys.executable, str(ROOT/'tools/submit.py'), str(root),
                                     '--partition', 'test-only'], capture_output=True, text=True, check=True)
            self.assertIn('Dry run only', result.stdout)
            self.assertIn('--gres=gpu:8', result.stdout)
            self.assertFalse((root/'submission.lock').exists())
            self.assertFalse((root/'allocation.out').exists())

    def test_capture_finds_announced_engine_ports(self):
        from shared.capture import engine_ports
        ip = '192.0.2.20'
        self.assertEqual(engine_ports('python -m sglang.launch_server --model-path /m --port 15000', ip), {15000})
        self.assertEqual(engine_ports(f'Launch HttpServerEngineAdapter at: {ip}:15001', ip), {15001})
        self.assertEqual(engine_ports(f"Ports for engine 0: {{'host': '{ip}', 'port': 15002}}", ip), {15002})
        self.assertEqual(engine_ports('Launch HttpServerEngineAdapter at: 10.0.0.1:15003', ip), set())


if __name__ == '__main__':
    unittest.main()


class CredentialTests(unittest.TestCase):
    def test_environment_by_default_and_profile_label_when_configured(self):
        from shared import credentials
        with tempfile.TemporaryDirectory() as temp:
            profile = Path(temp)/'profile'
            profile.write_text('export ORG_KEY="from-profile"\n')
            self.assertEqual(credentials.local_config('missing', Path(temp)/'absent.json'), {})
            with unittest.mock.patch.dict('os.environ', {'DEMO_API_KEY': 'from-env'}):
                self.assertEqual(credentials.api_key('DEMO_API_KEY'), 'from-env')
                with unittest.mock.patch.object(credentials, 'PROFILE', profile):
                    self.assertEqual(credentials.profile_secret('ORG_KEY', profile), 'from-profile')
            with unittest.mock.patch.dict('os.environ', {}, clear=True), self.assertRaises(RuntimeError):
                credentials.api_key('DEMO_API_KEY')


@unittest.skipUnless(importlib.util.find_spec('reasoning_gym'), 'needs reasoning-gym==0.1.25')
class ScoringTests(unittest.TestCase):
    def test_answer_after_thinking_is_required(self):
        from shared import scoring
        for domain in prepare.recipe.DOMAINS:
            label = json.loads(prepare.puzzles.packaged_bytes(prepare.recipe.data_file(domain, 'dev')).splitlines()[0])['label']
            answer = f"<answer>{json.loads(label)['answer']}</answer>"
            self.assertEqual(scoring.score('reasoning</think>' + answer, label), 1.0)
            self.assertEqual(scoring.score(answer, label), 0.0)  # Cut off before the thought ended.
            self.assertEqual(scoring.score('reasoning</think><answer>wrong</answer>', label), 0.0)
            self.assertEqual(scoring.score_reply(answer, label), 1.0)


class ThinkingTests(unittest.TestCase):
    """Every framework's generated launch configuration trains and evaluates with thinking on."""

    def load(self, framework, module):
        path = ROOT/'frameworks'/framework/f'{module}.py'
        sys.path.insert(0, str(path.parent))
        self.addCleanup(sys.path.remove, str(path.parent))
        spec = importlib.util.spec_from_file_location(f'{framework}_{module}', path)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded

    def test_thinking_on_everywhere(self):
        site = json.loads((ROOT/'config/site.example.json').read_text())
        thinking = json.dumps({'enable_thinking': True})
        with tempfile.TemporaryDirectory() as temp:
            result = Path(temp)
            prime = self.load('prime-rl', 'make_config').recipe_input(result, site)['orchestrator']
            self.assertTrue(prime['renderer']['enable_thinking'])
            self.assertTrue(prime['eval']['sampling']['extra_body']['chat_template_kwargs']['enable_thinking'])
            slime = self.load('slime', 'train')
            with unittest.mock.patch.object(slime, 'site', return_value=site), \
                    unittest.mock.patch.object(slime, 'model_args', return_value=[]):
                argv = slime.build(result)
            kwargs = argv[argv.index('--apply-chat-template-kwargs') + 1]
            self.assertEqual(json.loads(kwargs), {'enable_thinking': True})
            miles = self.load('miles', 'node').MilesNode.__new__(self.load('miles', 'node').MilesNode)
            miles.site, miles.result = site, result
            with unittest.mock.patch.dict('os.environ', {'CAMPAIGN_PYDEPS': '/pydeps', 'PYTHONPATH': ''}):
                command, _ = miles.train_command({d: 'http://teacher' for d in prepare.recipe.DOMAINS})
            extra = shlex.split(command[command.index('--extra-args') + 1])
            self.assertEqual(json.loads(extra[extra.index('--apply-chat-template-kwargs') + 1]), json.loads(thinking))
            verl = self.load('verl', 'train')
            with unittest.mock.patch.object(verl, 'site', return_value=site):
                verl_args = verl.overrides(result, result/'train.parquet', [result/'dev.parquet'])
            self.assertIn('+data.apply_chat_template_kwargs.enable_thinking=True', verl_args)
