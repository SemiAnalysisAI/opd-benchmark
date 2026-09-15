"""Prepare one isolated campaign from pinned source and recorded launch templates."""
import argparse
import gzip
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
FRAMEWORKS = ('miles', 'prime-rl', 'slime')
REQUIRED = ('assets', 'base_model', 'megatron_model', 'teachers', 'container_image',
            'megatron_source', 'uv', 'generation_node', 'trainer_node', 'generation_ip', 'trainer_ip')


def load_site(path):
    site = json.loads(Path(path).read_text())
    missing = set(REQUIRED) - set(site)
    if missing:
        raise ValueError(f'Missing site fields: {sorted(missing)}')
    for key in REQUIRED:
        value = site[key]
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_./:+-]+', value):
            raise ValueError(f'{key} must contain only safe path, hostname or IP characters')
    for key in REQUIRED[:7]:
        if not Path(site[key]).is_absolute():
            raise ValueError(f'{key} must be an absolute path')
    if site['generation_node'] == site['trainer_node']:
        raise ValueError('The two roles need different nodes')
    if site['generation_node'] > site['trainer_node']:
        raise ValueError('The recorded controller requires the generation node first in Slurm hostname order')
    gen, trainer = (ipaddress.IPv4Address(site[k]) for k in ('generation_ip', 'trainer_ip'))
    if trainer >= gen:
        raise ValueError('The recorded Ray placement requires trainer IP < generation IP')
    for key in ('base_model', 'megatron_model', 'teachers'):
        if not Path(site[key]).is_relative_to(Path(site['assets'])):
            raise ValueError(f'{key} must be within assets so the container can access it')
    return site


def substitutions(site, output):
    values = {k.upper(): v for k, v in site.items()}
    values.update(CAMPAIGN=str(output), BASE_PARENT=str(Path(site['base_model']).parent),
                  MEGATRON_PARENT=str(Path(site['megatron_model']).parent),
                  GENERATION_IP_REGEX=site['generation_ip'].replace('.', r'\.'))
    return values


def render(text, values):
    for name, value in values.items():
        text = text.replace(f'@{name}@', value)
    remaining = re.findall(r'@[A-Z_]+@', text)
    if remaining:
        raise ValueError(f'Unresolved template values: {remaining}')
    return text


def materialize_data(destination):
    destination.mkdir()
    manifest = json.loads((REPO/'data/manifest.json').read_text())
    for filename, record in manifest.items():
        data = gzip.decompress((REPO/'data'/f'{filename}.gz').read_bytes())
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError(f'Dataset checksum mismatch: {filename}')
        (destination/filename).write_bytes(data)


def prepare(framework, site, output, source_cache=None):
    output = output.resolve()
    if not re.fullmatch(r'[A-Za-z0-9_./+-]+', str(output)):
        raise ValueError('The campaign path cannot contain whitespace or shell metacharacters')
    if output.exists():
        raise ValueError('The campaign directory already exists. Select a new empty path.')
    package = REPO/'frameworks'/framework
    manifest = json.loads((package/'manifest.json').read_text())
    patch = package/'source.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest['source_patch_sha256']:
        raise ValueError('Source patch checksum mismatch')
    output.mkdir(parents=True)
    source = output/'source'
    clone_from = str(source_cache) if source_cache else manifest['repository']
    subprocess.run(['git', 'clone', '--no-checkout', clone_from, str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'checkout', '--detach', manifest['revision']], check=True)
    if framework == 'prime-rl':
        subprocess.run(['git', '-c', 'url.https://github.com/.insteadOf=git@github.com:',
                        '-C', str(source), 'submodule', 'update', '--init', '--recursive'], check=True)
    subprocess.run(['git', '-C', str(source), 'apply', '--check', str(patch)], check=True)
    subprocess.run(['git', '-C', str(source), 'apply', str(patch)], check=True)
    values = substitutions(site, output)
    for entry in sorted((package/'campaign').rglob('*')):
        if not entry.is_file():
            continue
        target = output/entry.relative_to(package/'campaign')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(entry.read_text(), values))
    materialize_data(output/'data')
    shutil.copyfile(REPO/'tools/submit.py', output/'launch.py')
    (output/'models').mkdir()
    (output/'models/Qwen3.6-35B-A3B').symlink_to(site['base_model'])
    (output/'models/Qwen3.6-35B-A3B_torch_dist').symlink_to(site['megatron_model'])
    (output/'teachers').symlink_to(site['teachers'])
    (output/'site.json').write_text(json.dumps(site, indent=2)+'\n')
    (output/'package-provenance.json').write_text(json.dumps({'framework':framework, **manifest}, indent=2)+'\n')
    print(f'Prepared {output}. No GPU job was submitted. Complete docs/SETUP.md before submission.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('framework', choices=FRAMEWORKS)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-cache', type=Path, help='Optional local Git clone; the recorded revision is still enforced')
    args = parser.parse_args()
    prepare(args.framework, load_site(args.site), args.output, args.source_cache)


if __name__ == '__main__':
    main()
