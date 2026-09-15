import ast
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare', ROOT/'tools/prepare.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PackageTests(unittest.TestCase):
    def test_all_templates_render_and_compile(self):
        site = prepare.load_site(ROOT/'site.example.json')
        values = prepare.substitutions(site, Path('/shared/test-campaign'))
        for path in (ROOT/'frameworks').glob('*/campaign/**/*'):
            if path.is_file():
                text = prepare.render(path.read_text(), values)
                self.assertNotIn('/Users/joey', text, str(path))
                self.assertNotIn('clustermax-campaigns', text, str(path))
                if path.suffix == '.py':
                    ast.parse(text, filename=str(path))
                elif path.suffix == '.json':
                    json.loads(text)

    def test_dataset_integrity_and_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)/'data'
            prepare.materialize_data(dest)
            manifest = json.loads((ROOT/'data/manifest.json').read_text())
            for name, entry in manifest.items():
                rows = [json.loads(line) for line in (dest/name).read_text().splitlines()]
                self.assertEqual(len(rows), entry['rows'])
            self.assertEqual(manifest['mixed-train.jsonl']['rows'], 20000)
            self.assertEqual(manifest['countdown4-dev.jsonl']['rows'], 512)
            self.assertEqual(manifest['graph12-dev.jsonl']['rows'], 512)

    def test_patch_integrity(self):
        for name in prepare.FRAMEWORKS:
            root = ROOT/'frameworks'/name
            manifest = json.loads((root/'manifest.json').read_text())
            self.assertEqual(hashlib.sha256((root/'source.patch').read_bytes()).hexdigest(), manifest['source_patch_sha256'])
            self.assertEqual(len(manifest['revision']), 40)

    def test_invalid_site_and_placeholders_fail(self):
        with self.assertRaises(ValueError):
            prepare.render('@MISSING@', {})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad.json'
            path.write_text('{}')
            with self.assertRaises(ValueError):
                prepare.load_site(path)
            site = json.loads((ROOT/'site.example.json').read_text())
            site['base_model'] = '/tmp/model\";print(1)'
            path.write_text(json.dumps(site))
            with self.assertRaises(ValueError):
                prepare.load_site(path)

    def test_existing_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                prepare.prepare('slime', prepare.load_site(ROOT/'site.example.json'), Path(temp))

    def test_submit_defaults_to_printing_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'site.json').write_text((ROOT/'site.example.json').read_text())
            (root/'package-provenance.json').write_text('{"framework":"slime"}')
            result = subprocess.run([sys.executable, str(ROOT/'tools/submit.py'), str(root),
                                     '--partition', 'test-only'], capture_output=True, text=True, check=True)
            self.assertIn('Dry run only', result.stdout)
            self.assertIn('--gres=gpu:8', result.stdout)
            self.assertFalse((root/'submission.lock').exists())
            self.assertFalse((root/'allocation.out').exists())


if __name__ == '__main__':
    unittest.main()
