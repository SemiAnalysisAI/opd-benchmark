"""Check that each framework's source.patch reproduces the recorded source.

    python3 tools/verify_sources.py [--miles CLONE] [--prime-rl CLONE] [--slime CLONE]

For each local upstream clone given, applies the patch to a `git archive` of the
pinned revision and compares every changed file with its recorded SHA-256.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[1]


def verify(name, repository):
    package = REPO/'frameworks'/name/'upstream'
    manifest = json.loads((package/'manifest.json').read_text())
    raw = subprocess.check_output(['git', '-C', str(repository), 'archive', manifest['revision']])
    with tempfile.TemporaryDirectory(prefix='opd-source-check-') as temp:
        root = Path(temp)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            archive.extractall(root, filter='data')
        subprocess.run(['git', 'apply', '--check', str(package/'source.patch')], cwd=root, check=True)
        subprocess.run(['git', 'apply', str(package/'source.patch')], cwd=root, check=True)
        for path, expected in manifest['patched_source_sha256'].items():
            actual = hashlib.sha256((root/path).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'{name}/{path} differs from the recorded source')
    print(f'{name}: patch applied and all {len(manifest["patched_source_sha256"])} changed files match the recorded source')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('miles', 'prime-rl', 'slime'):  # verl and NeMo-RL run unmodified.
        parser.add_argument('--'+name, type=Path, metavar='CLONE', help=f'local {name} Git clone')
    args = vars(parser.parse_args())
    if not any(args.values()):
        parser.error('Provide at least one local upstream Git clone')
    for name, path in args.items():
        if path:
            verify(name.replace('_', '-'), path)


if __name__ == '__main__':
    main()
