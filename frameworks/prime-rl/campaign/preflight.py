"""Verify puzzle identities, scoring, and tokenization before GPU training."""

import hashlib
import json
from pathlib import Path
import struct

from transformers import AutoTokenizer
import torch
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import get_model
from renderers import Qwen36RendererConfig, create_renderer
from mopd_puzzles.taskset import MopdPuzzlesConfig, MopdPuzzlesTaskset
from mopd_puzzles.scoring import score

ROOT = Path(__file__).resolve().parent
BASE = '@BASE_MODEL@'
model_config = ModelConfig(name=BASE, impl='custom', attn='flash_attention_4',
    optimization_dtype='bfloat16', reduce_dtype='bfloat16',
    vlm={'vision_encoder_attr': 'model.visual', 'language_model_attr': 'model.language_model',
         'freeze_vision_encoder': True})
meta_model = get_model(model_config, device=torch.device('meta'))
assert meta_model.model.visual.config._attn_implementation == 'sdpa'
assert meta_model.model.language_model.config._attn_implementation == 'flash_attention_4'
assert all(parameter.device.type == 'meta' for parameter in meta_model.parameters())
print(json.dumps({'model_setup': 'passed', 'vision_attention': 'sdpa', 'text_attention': 'flash_attention_4',
                  'allocation': 'meta, no model weights loaded'}))
del meta_model
cache = Path(BASE) / 'prime'
assert (cache / '.prime-v1').is_file(), 'The previously completed conversion is required; do not modify the original model cache'
index = json.loads((cache / 'model.safetensors.index.json').read_text())['weight_map']
for filename in set(index.values()):
    shard = cache / filename
    with shard.open('rb') as stream:
        header_length = struct.unpack('<Q', stream.read(8))[0]
        assert 0 < header_length < 16 * 1024 * 1024
        header = json.loads(stream.read(header_length))
    expected_keys = {key for key, value in index.items() if value == filename}
    assert expected_keys == set(header) - {'__metadata__'}, filename
    data_length = max(header[key]['data_offsets'][1] for key in expected_keys)
    assert shard.stat().st_size == 8 + header_length + data_length, filename
print(json.dumps({'verified_conversion_cache': str(cache), 'shards': len(set(index.values())), 'tensors': len(index)}))
tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
renderer = create_renderer(tokenizer, Qwen36RendererConfig(enable_thinking=False))
assert score('<answer>1+2+3+4</answer>', {'domain': 'countdown', 'numbers': [1, 2, 3, 4], 'target': 10}) == 1
assert score('<answer>1+2</answer>', {'domain': 'countdown', 'numbers': [1, 2, 3, 4], 'target': 10}) == 0
puzzle = {'domain': 'graph_color', 'puzzle': {'vertices': [0, 1], 'edges': [[0, 1]], 'color_options': [1, 2]}}
assert score('<answer>{"0":1,"1":2}</answer>', puzzle) == 1
assert score('<answer>{"0":1,"1":1}</answer>', puzzle) == 0
summary = {}
for domain, prefix in [('countdown', 'countdown4'), ('graph_color', 'graph12')]:
    for split, count in [('train', 10000), ('dev', 512)]:
        path = ROOT / 'data' / f'{prefix}-{split}.jsonl'
        taskset = MopdPuzzlesTaskset(MopdPuzzlesConfig(data_path=str(path), domain=domain))
        tasks = taskset.load()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(tasks) == len(rows) == count
        for i, (task, row) in enumerate(zip(tasks, rows, strict=True)):
            assert task.data.system_prompt == row['prompt'][0]['content']
            assert task.data.prompt == row['prompt'][1]['content']
            assert task.data.label == row['label']
            if i < 20:
                native = renderer.render_ids(row['prompt'], add_generation_prompt=True)
                expected = tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True, enable_thinking=False)
                if hasattr(expected, 'input_ids'):
                    expected = expected.input_ids
                assert native == expected, (domain, split, i, native[-12:], expected[-12:])
                assert len(native) + 256 <= 2048
        summary[path.name] = {'rows': count, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'tokenization_samples': 20}
print(json.dumps({'status': 'passed', 'data': summary, 'scoring_cases': 4}, indent=2))
