"""Read the packaged puzzle datasets and check their recorded checksums.

The checkout stores each dataset as `data/<name>.gz` next to `data/manifest.json`.
A prepared campaign holds the verified, uncompressed files in its own `data/`.
"""
import gzip
import hashlib
import json
from pathlib import Path

from shared import recipe

REPO_DATA = Path(__file__).resolve().parents[1] / 'data'


def manifest(data_dir=REPO_DATA):
    return json.loads((Path(data_dir) / 'manifest.json').read_text())


def packaged_bytes(name, data_dir=REPO_DATA):
    """Uncompressed bytes of one packaged dataset, after checking its size and SHA-256."""
    record = manifest(data_dir)[name]
    data = gzip.decompress((Path(data_dir) / f'{name}.gz').read_bytes())
    if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError(f'Dataset checksum mismatch: {name}')
    return data


def load_split(domain, split, data_dir=REPO_DATA):
    """Rows of one packaged split. Each row gains `_index`, its position in the file."""
    name = recipe.data_file(domain, split)
    rows = [json.loads(line) for line in packaged_bytes(name, data_dir).splitlines()]
    if len(rows) != manifest(data_dir)[name]['rows']:
        raise ValueError(f'Dataset row count mismatch: {name}')
    for index, row in enumerate(rows):
        if row['metadata']['domain'] != domain:
            raise ValueError(f'{name} row {index} belongs to {row["metadata"]["domain"]}')
        row['_index'] = index
    return rows
