"""Check that a Tinker-compatible server supports every call sft.py, rl.py, opd.py and mopd.py make.

    TINKER_BASE_URL=http://HOST:PORT python hosted/selfhosted/smoke.py --backend skyrl --output /abs/smoke.json

Steps, each timed and recorded: capabilities; a fresh LoRA; sampling with logprobs, a stop sequence and a seed;
prefill logprobs (compute_logprobs); cross_entropy, importance_sampling and ppo forward_backward on a
--long-tokens datum; optim_step; save_state; save_weights_for_sampler and sampling from that path; restoring
from the state with its optimizer. `--teacher PATH` also samples a sampler path saved by an earlier process,
which is how opd.py and mopd.py use SFT teachers. The first failing step stops the check and is recorded.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'opd'))
import tinker  # noqa: E402

from backend import BACKENDS  # noqa: E402
from common import DEFAULT_MODEL, MODELS, load_renderer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backend', choices=sorted(BACKENDS), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rank', type=int, default=64)
    p.add_argument('--long-tokens', type=int, default=30_000, help='Length of the training datum')
    p.add_argument('--teacher', help='A sampler path saved by another process')
    args = p.parse_args()
    model = MODELS[DEFAULT_MODEL]
    tokenizer, renderer = load_renderer(model)
    record = {'backend': args.backend, 'steps': []}

    def step(name, fn):
        start = time.monotonic()
        try:
            detail = fn()
        except Exception as error:
            record['steps'].append({'step': name, 'ok': False, 'seconds': time.monotonic() - start,
                                    'error': f'{type(error).__name__}: {error}'[:2000], 'traceback': traceback.format_exc()})
            args.output.write_text(json.dumps(record, indent=2))
            raise SystemExit(f'FAILED {name}: {error}')
        record['steps'].append({'step': name, 'ok': True, 'seconds': time.monotonic() - start, **(detail or {})})
        print(json.dumps(record['steps'][-1]), flush=True)
        args.output.write_text(json.dumps(record, indent=2))

    state = {}
    messages = [{'role': 'user', 'content': 'What is 17 * 23? Answer with <answer>N</answer>.'}]
    prompt = tinker.ModelInput.from_ints(renderer.build_generation_prompt(messages).to_ints())
    params = tinker.SamplingParams(max_tokens=256, temperature=1.0, top_p=1.0, top_k=-1,
                                   stop=renderer.get_stop_sequences(), seed=7)

    def connect():
        state['backend'] = BACKENDS[args.backend](experiment='smoke')
        record['account'] = state['backend'].account
        return {'models': [m['model_name'] for m in state['backend'].capabilities(model[0])['supported_models']]}

    def new_lora():
        state['train'] = state['backend'].new_student(model[0], args.rank, 20260921, {'purpose': 'smoke'})
        try:
            return {'info': json.loads(state['train'].get_info().model_dump_json())}
        except Exception as error:
            return {'info_unavailable': str(error)[:300]}

    def sample(client_key, n=2):
        def fn():
            client = state[client_key]
            result = client.sample(prompt, num_samples=n, sampling_params=params).result()
            seqs = result.sequences
            assert len(seqs) == n and all(len(s.tokens) == len(s.logprobs) for s in seqs)
            return {'lengths': [len(s.tokens) for s in seqs], 'stop_reasons': [s.stop_reason for s in seqs],
                    'text': tokenizer.decode(seqs[0].tokens)[-200:]}
        return fn

    def base_sampler():
        state['sampler'] = state['backend'].sync(state['train'], 'smoke-000')[0]

    def prefill():
        ids = prompt.to_ints() + tokenizer.encode(' The answer is <answer>391</answer>.')
        lp = state['sampler'].compute_logprobs(tinker.ModelInput.from_ints(ids)).result()
        assert len(lp) == len(ids) and lp[0] is None and all(x is not None for x in lp[1:])
        return {'tokens': len(ids), 'mean_logprob': sum(lp[1:]) / (len(lp) - 1)}

    def long_datum(loss):
        n = args.long_tokens
        ids = (tokenizer.encode('The quick brown fox jumps over the lazy dog. ') * (n // 10 + 1))[:n]
        shape = [n - 1]
        if loss == 'cross_entropy':
            inputs = {'target_tokens': tinker.TensorData(data=ids[1:], dtype='int64', shape=shape),
                      'weights': tinker.TensorData(data=[1.0] * (n - 1), dtype='float32', shape=shape)}
        else:
            inputs = {'target_tokens': tinker.TensorData(data=ids[1:], dtype='int64', shape=shape),
                      'logprobs': tinker.TensorData(data=[-1.0] * (n - 1), dtype='float32', shape=shape),
                      'advantages': tinker.TensorData(data=[0.1] * (n - 1), dtype='float32', shape=shape)}
        return tinker.Datum(model_input=tinker.ModelInput.from_ints(ids[:-1]), loss_fn_inputs=inputs)

    def forward_backward(loss, config=None):
        def fn():
            out = state['train'].forward_backward([long_datum(loss)], loss_fn=loss, loss_fn_config=config).result()
            return {'metrics': out.metrics, 'outputs': sorted(out.loss_fn_outputs[0])}
        return fn

    def optim():
        out = state['train'].optim_step(tinker.AdamParams(learning_rate=1e-5, beta1=0.9, beta2=0.999, eps=1e-8,
                                                          weight_decay=0.0, grad_clip_norm=1.0)).result()
        return {'metrics': out.metrics}

    def save_state():
        state['state_path'] = state['train'].save_state(name='smoke-state').result().path
        return {'path': state['state_path']}

    def save_sampler():
        state['sampler'], path = state['backend'].sync(state['train'], 'smoke-001')
        return {'path': path}

    def restore():
        state['restored'] = state['backend'].resume_student(state['state_path'])
        state['restored_sampler'], path = state['backend'].sync(state['restored'], 'smoke-restored')
        return {'path': path}

    def teacher():
        state['teacher'] = state['backend'].sampler(path=args.teacher)
        return sample('teacher')()

    step('capabilities', connect)
    step('create_lora_training_client', new_lora)
    step('save_weights_for_sampler (initial)', base_sampler)
    step('sample (logprobs, stop, seed)', sample('sampler'))
    step('compute_logprobs', prefill)
    step(f'forward_backward cross_entropy ({args.long_tokens} tokens)', forward_backward('cross_entropy'))
    step(f'forward_backward importance_sampling ({args.long_tokens} tokens)', forward_backward('importance_sampling'))
    step(f'forward_backward ppo ({args.long_tokens} tokens)',
         forward_backward('ppo', {'clip_low_threshold': 0.8, 'clip_high_threshold': 1.2}))
    step('optim_step', optim)
    step('save_state', save_state)
    step('save_weights_for_sampler and sample', lambda: (save_sampler(), sample('sampler')())[1])
    step('create_training_client_from_state_with_optimizer', restore)
    step('sample restored', sample('restored_sampler'))
    if args.teacher:
        step('sample a sampler path from an earlier process', teacher)
    record['ok'] = True
    args.output.write_text(json.dumps(record, indent=2))
    print('SMOKE OK', flush=True)


if __name__ == '__main__':
    main()
