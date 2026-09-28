"""Prepare a self-contained campaign directory for one framework. Nothing is submitted.

The directory gets the upstream source at its pinned revision with the patch applied,
the verified datasets, a copy of `shared/`, the framework's campaign files, `launch.py`
(a copy of tools/submit.py), links to the models, and `site.json`, which campaign
processes read at runtime.
"""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from shared import puzzles, recipe  # noqa: E402

FRAMEWORKS = ('miles', 'nemo-rl', 'prime-rl', 'slime', 'verl')
PACKAGE = REPO / 'shared'
PATH_FIELDS = ('assets', 'base_model', 'megatron_model', 'teachers', 'container_image', 'megatron_source', 'uv')
HOST_FIELDS = ('generation_node', 'trainer_node', 'generation_ip', 'trainer_ip')
REQUIRED = PATH_FIELDS + HOST_FIELDS
# NeMo-RL's MOPD reserves a whole node for the teacher and runs in its own NGC container.
NEMO_RL_FIELDS = ('teacher_node', 'teacher_ip', 'nemo_rl_container')
SAFE_VALUE = r'[A-Za-z0-9_./:+-]+'


def load_site(path, framework='miles'):
    site = json.loads(Path(path).read_text())
    required = REQUIRED + (NEMO_RL_FIELDS if framework == 'nemo-rl' else ())
    missing = set(required) - set(site)
    if missing:
        raise ValueError(f'Missing site fields: {sorted(missing)}')
    for key in required:
        value = site[key]
        if not isinstance(value, str) or not re.fullmatch(SAFE_VALUE, value):
            raise ValueError(f'{key} must contain only safe path, hostname or IP characters')
    for key in PATH_FIELDS + (('nemo_rl_container',) if framework == 'nemo-rl' else ()):
        if not Path(site[key]).is_absolute():
            raise ValueError(f'{key} must be an absolute path')
    nodes = [site['generation_node'], site['trainer_node'], *([site['teacher_node']] if framework == 'nemo-rl' else [])]
    if len(set(nodes)) != len(nodes):
        raise ValueError('Every role needs its own node')
    generation, trainer = (ipaddress.IPv4Address(site[k]) for k in ('generation_ip', 'trainer_ip'))
    # Miles and Slime sort Ray bundles by node IP; Prime-RL uses explicit addresses and verl places its own pools.
    if framework in ('miles', 'slime') and trainer >= generation:
        raise ValueError('The recorded Ray placement requires trainer IP < generation IP')
    for key in ('base_model', 'megatron_model', 'teachers'):
        if not Path(site[key]).is_relative_to(Path(site['assets'])):
            raise ValueError(f'{key} must be within assets so the container can access it')
    return site


def materialize_data(destination):
    destination.mkdir()
    for filename in puzzles.manifest():
        (destination / filename).write_bytes(puzzles.packaged_bytes(filename))


def campaign_files(framework):
    """(source, target relative to the campaign) for `shared/` and the framework's campaign files."""
    files = [(p, Path('shared') / p.name) for p in sorted(PACKAGE.glob('*.py'))]
    framework_dir = REPO / 'frameworks' / framework
    return files + [(p, p.relative_to(framework_dir)) for p in sorted(framework_dir.rglob('*'))
                    if p.is_file() and not {'upstream', '__pycache__'} & set(p.relative_to(framework_dir).parts)]


def prepare(framework, site, output, source_cache=None, domains=None):
    output = output.resolve()
    if not re.fullmatch(SAFE_VALUE, str(output)):
        raise ValueError('The campaign path cannot contain whitespace or shell metacharacters')
    if output.exists():
        raise ValueError('The campaign directory already exists. Select a new empty path.')
    upstream = REPO / 'frameworks' / framework / 'upstream'
    manifest = json.loads((upstream / 'manifest.json').read_text())
    patch = upstream / 'source.patch'  # Absent when the framework runs unmodified.
    if patch.exists() and hashlib.sha256(patch.read_bytes()).hexdigest() != manifest['source_patch_sha256']:
        raise ValueError('Source patch checksum mismatch')
    output.mkdir(parents=True)
    source = output / 'source'
    clone_from = str(source_cache) if source_cache else manifest['repository']
    subprocess.run(['git', 'clone', '--no-checkout', clone_from, str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'checkout', '--detach', manifest['revision']], check=True)
    if framework == 'prime-rl':
        subprocess.run(['git', '-c', 'url.https://github.com/.insteadOf=git@github.com:',
                        '-C', str(source), 'submodule', 'update', '--init', '--recursive'], check=True)
    if patch.exists():
        subprocess.run(['git', '-C', str(source), 'apply', '--check', str(patch)], check=True)
        subprocess.run(['git', '-C', str(source), 'apply', str(patch)], check=True)
    for source_file, target in campaign_files(framework):
        target = output / target
        target.parent.mkdir(parents=True, exist_ok=True)
        # copyfile follows symlinks, so Prime-RL's link to the scorer becomes a regular file.
        shutil.copyfile(source_file, target)
    materialize_data(output / 'data')
    shutil.copyfile(REPO / 'tools/submit.py', output / 'launch.py')
    (output / 'models').mkdir()
    (output / 'models' / recipe.BASE_MODEL_DIR).symlink_to(site['base_model'])
    (output / 'models' / recipe.MEGATRON_MODEL_DIR).symlink_to(site['megatron_model'])
    (output / 'teachers').symlink_to(site['teachers'])
    (output / 'site.json').write_text(json.dumps(site, indent=2) + '\n')
    if domains:  # Read by the campaign's `shared/recipe.py`.
        (output / 'experiment.json').write_text(json.dumps({'domains': list(domains)}, indent=2) + '\n')
    (output / 'package-provenance.json').write_text(json.dumps({'framework': framework, **manifest}, indent=2) + '\n')
    print(f'Prepared {output}. No GPU job was submitted. Complete docs/SETUP.md before submission.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('framework', choices=FRAMEWORKS)
    parser.add_argument('--site', type=Path, default=REPO / 'config/site.local.json',
                        help='site file (default: config/site.local.json)')
    parser.add_argument('--output', type=Path, required=True, help='campaign directory; must not exist')
    parser.add_argument('--source-cache', type=Path, help='local Git clone to clone from; the pinned revision still applies')
    parser.add_argument('--domains', nargs='+', choices=recipe.ALL_DOMAINS,
                        help='one task for single-teacher OPD, e.g. caesar_cipher (default: all)')
    args = parser.parse_args()
    if args.domains and len(args.domains) != 1 and tuple(args.domains) != recipe.ALL_DOMAINS:
        parser.error(f'--domains takes one task or all of {recipe.ALL_DOMAINS} in order')
    prepare(args.framework, load_site(args.site, args.framework), args.output, args.source_cache, args.domains)


if __name__ == '__main__':
    main()
