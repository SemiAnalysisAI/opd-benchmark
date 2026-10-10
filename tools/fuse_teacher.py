"""Rewrite a downloaded teacher into the base checkpoint's fused-expert layout.

    python3 tools/fuse_teacher.py TEACHER_DIR OUTPUT_DIR BASE_MODEL_DIR

The GRPO teachers on the Hub are Prime-RL weight broadcasts: each MoE expert is its
own module (`experts.N.{gate,up,down}_proj.weight`) and the untrained MTP head is
absent. The base Qwen3.6 checkpoint fuses each layer's experts into
`experts.gate_up_proj` [E, 2I, H] and `experts.down_proj` [E, H, I]. The output has
exactly the base's tensors: the teacher's weights with experts re-fused, plus the
base's MTP head. The teacher's other files (config, tokenizer) are copied as is.
Needs torch and safetensors.
"""
from collections import defaultdict
import json
from pathlib import Path
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT = re.compile(r'^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$')
SHARD_BYTES = 5 * 1024**3


def fuse(src, dst, base):
    src_map = json.loads((src / 'model.safetensors.index.json').read_text())['weight_map']
    base_map = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
    handles = {}

    def tensor(root, mapping, key):
        if (root, mapping[key]) not in handles:
            handles[root, mapping[key]] = safe_open(str(root / mapping[key]), 'pt')
        return handles[root, mapping[key]].get_tensor(key)

    experts = defaultdict(dict)
    for key in src_map:
        if match := EXPERT.match(key):
            experts[match.group(1)][int(match.group(2)), match.group(3)] = key
    used, shards, buffer, size, total = set(), [], {}, 0, 0
    dst.mkdir(parents=True, exist_ok=False)

    def flush():
        nonlocal buffer, size
        if buffer:
            name = f'model-{len(shards) + 1:05d}.safetensors'
            save_file(buffer, str(dst / name), metadata={'format': 'pt'})
            shards.append((name, list(buffer)))
            buffer, size = {}, 0

    for key in sorted(base_map):  # Base order keeps related tensors together.
        if key.startswith('mtp.'):
            value = tensor(base, base_map, key)  # Not trained by the teacher run; keep the base's.
        elif key in src_map:
            value = tensor(src, src_map, key)
            used.add(key)
        elif key.endswith(('.mlp.experts.gate_up_proj', '.mlp.experts.down_proj')):
            prefix, projection = key.rsplit('.', 1)
            group = experts[prefix]
            count = 1 + max(i for i, _ in group)
            parts = ('gate_proj', 'up_proj') if projection == 'gate_up_proj' else ('down_proj',)
            stacked = [torch.stack([tensor(src, src_map, group[i, part]) for i in range(count)]) for part in parts]
            value = torch.cat(stacked, dim=1)
            used.update(group[i, part] for i in range(count) for part in parts)
        else:
            raise KeyError(f'Base tensor {key} has no source in the teacher')
        expected = handles.get((base, base_map[key])) or safe_open(str(base / base_map[key]), 'pt')
        if list(value.shape) != list(expected.get_slice(key).get_shape()):
            raise ValueError(f'{key}: shape {tuple(value.shape)} differs from the base')
        buffer[key] = value.contiguous()
        size += value.numel() * value.element_size()
        total += value.numel() * value.element_size()
        if size >= SHARD_BYTES:
            flush()
    flush()
    if unused := set(src_map) - used:
        raise KeyError(f'{len(unused)} teacher tensors unused, for example {sorted(unused)[:3]}')
    weight_map = {}
    for i, (name, keys) in enumerate(shards, 1):
        final = f'model-{i:05d}-of-{len(shards):05d}.safetensors'
        (dst / name).rename(dst / final)
        weight_map.update({k: final for k in keys})
    (dst / 'model.safetensors.index.json').write_text(
        json.dumps({'metadata': {'total_size': total}, 'weight_map': weight_map}, indent=2))
    for path in src.iterdir():
        if path.is_file() and not path.name.endswith('.safetensors') and not (dst / path.name).exists():
            shutil.copy2(path, dst / path.name)
    return {'tensors': len(weight_map), 'shards': len(shards), 'bytes': total}


if __name__ == '__main__':
    print(json.dumps(fuse(*(Path(arg) for arg in sys.argv[1:4]))))
