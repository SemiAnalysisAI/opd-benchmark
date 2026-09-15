"""Serve frozen teachers on separate GPUs and validate prefill scoring."""
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

campaign = Path(os.environ['MOPD_CAMPAIGN'])
result = Path(os.environ['MOPD_RESULT'])
processes = []

def request(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read()
    return data

def stop(*_):
    raise SystemExit(1)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
try:
    from sglang.srt.managers.io_struct import GenerateReqInput
    (result / 'scoring-api.json').write_text(json.dumps({
        'sparse_field_present': 'token_ids_logprob_positions' in GenerateReqInput.__dataclass_fields__,
        'selected_mode': 'dense requested IDs',
        'reason': 'Use documented compatible API and validate returned requested-token scores.'
    }, indent=2))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(campaign / 'models/Qwen3.6-35B-A3B'), trust_remote_code=True)
    for gpu, domain in enumerate(('countdown', 'graph_color')):
        command = [sys.executable, '-m', 'sglang.launch_server',
                   '--model-path', str(campaign / 'teachers' / domain),
                   '--tokenizer-path', str(campaign / 'models/Qwen3.6-35B-A3B'),
                   '--host', '0.0.0.0', '--port', str(30000 + gpu), '--tp-size', '1',
                   '--context-length', '2048', '--mem-fraction-static', '0.8',
                   '--max-running-requests', '128', '--disable-cuda-graph',
                   '--chunked-prefill-size', '-1', '--disable-radix-cache',
                   '--prefill-max-requests', '1', '--moe-runner-backend', 'triton',
                   '--enable-metrics']
        (result / f'{domain}-command.json').write_text(json.dumps(command, indent=2))
        log = open(result / f'{domain}-server.out', 'wb')
        processes.append(subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
    deadline = time.monotonic() + 600
    for gpu, domain in enumerate(('countdown', 'graph_color')):
        url = f'http://127.0.0.1:{30000 + gpu}'
        while True:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError('A teacher process exited during startup')
            try:
                request(url + '/health')
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise TimeoutError('Teacher readiness exceeded ten minutes')
                time.sleep(2)
        name = 'countdown4' if domain == 'countdown' else 'graph12'
        row = json.loads((campaign / f'data/{name}-train.jsonl').open().readline())
        prompt = row['prompt']
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt}]
        encoded = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        ids = list(encoded['input_ids'] if hasattr(encoded, 'keys') else encoded)
        assert ids and all(isinstance(token, int) for token in ids), 'Expected one flat token sequence'
        # This synthetic suffix checks the scoring API. It is not an accuracy or throughput result.
        ids += tokenizer.encode('<answer>test</answer>', add_special_tokens=False)
        selected = sorted(set(ids[-16:]))
        payload = {'input_ids': ids, 'sampling_params': {'temperature': 0, 'max_new_tokens': 0},
                   'return_logprob': True, 'logprob_start_len': 0, 'token_ids_logprob': selected}
        (result / f'{domain}-probe-request.json').write_text(json.dumps(payload))
        timings = []
        for attempt in range(2):
            start = time.monotonic()
            raw = request(url + '/generate', payload)
            timings.append(time.monotonic() - start)
            response = json.loads(raw)
            (result / f'{domain}-probe-response-{attempt}.json').write_bytes(raw)
            meta = response['meta_info']
            assert meta.get('completion_tokens', 0) == 0, meta
            input_scores = meta['input_token_logprobs'][1:]
            assert len(input_scores) == len(ids) - 1
            assert all(math.isfinite(float(x[0])) for x in input_scores)
            scores = meta['input_token_ids_logprobs'][-1]
            mapping = {int(x[1]): float(x[0]) for x in scores}
            assert all(t in mapping and math.isfinite(mapping[t]) for t in selected), scores
        (result / f'{domain}-probe-timing.json').write_text(json.dumps({
            'input_tokens': len(ids), 'selected_ids': len(selected), 'seconds': timings,
            'purpose': 'API smoke test; synthetic suffix; not representative throughput'
        }, indent=2))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(('8.8.8.8', 80))
    ip = sock.getsockname()[0]
    sock.close()
    (result / 'teachers-ready.json').write_text(json.dumps({'ip': ip, 'ports': [30000, 30001]}))
    while True:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('A frozen teacher exited; stop the student allocation')
        time.sleep(2)
finally:
    for p in processes:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for p in processes:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
