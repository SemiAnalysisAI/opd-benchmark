"""Benchmark one Hugging Face checkpoint on the packaged dev sets with vLLM, thinking on.

    python tools/benchmark.py MODEL OUTPUT_DIR [--domains ...] [--samples 8] [--seed 0] [--gpus 8]

Each dev prompt gets `--samples` responses at the training sampling settings:
temperature 1, top-p 1, up to `recipe.MAX_RESPONSE_TOKENS` tokens. Prompts are split
across `--gpus` single-GPU vLLM engines. The answer is read after the last `</think>`,
so a response cut off mid-thought scores 0.

Writes `responses.jsonl.gz` and `summary.json` (avg@k, pass@k, finished-thinking and
truncation rates, response lengths, output tokens per second). Needs vLLM, plus
reasoning-gym 0.1.25 installed or in `$CAMPAIGN_PYDEPS`.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from shared import puzzles, recipe  # noqa: E402


def prompts(domains):
    return [(domain, row) for domain in domains for row in puzzles.load_split(domain, 'dev')]


def worker(args):
    from vllm import LLM, SamplingParams

    mine = prompts(args.domains)[args.worker::args.gpus]
    # Text only: some trainers export the language model without the unused vision tower.
    llm = LLM(model=str(args.model), tensor_parallel_size=1, max_model_len=recipe.CONTEXT_LENGTH, seed=args.seed,
              language_model_only=True)
    params = SamplingParams(n=args.samples, temperature=1.0, top_p=1.0, max_tokens=recipe.MAX_RESPONSE_TOKENS,
                            seed=args.seed)
    started = time.time()
    outputs = llm.chat([row['prompt'] for _, row in mine], params,
                       chat_template_kwargs={'enable_thinking': recipe.ENABLE_THINKING})
    seconds = time.time() - started
    with gzip.open(args.output / f'worker-{args.worker}.jsonl.gz', 'wt') as stream:
        for (domain, row), output in zip(mine, outputs):
            for sample, completion in enumerate(output.outputs):
                stream.write(json.dumps({'domain': domain, 'index': row['_index'], 'sample': sample,
                                         'label': row['label'], 'text': completion.text,
                                         'response_tokens': len(completion.token_ids),
                                         'finish_reason': completion.finish_reason, 'worker_seconds': seconds}) + '\n')


def summarize(args):
    from shared.scoring import score

    responses = []
    for path in sorted(args.output.glob('worker-*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            responses += [json.loads(line) for line in stream]
    for response in responses:
        response['score'] = score(response['text'], response.pop('label'))
    seconds = max(r['worker_seconds'] for r in responses)
    summary = {'model': str(args.model), 'samples': args.samples, 'seed': args.seed, 'gpus': args.gpus,
               'generation_seconds': seconds,
               'output_tokens_per_second': sum(r['response_tokens'] for r in responses) / seconds, 'domains': {}}
    for domain in args.domains:
        mine = [r for r in responses if r['domain'] == domain]
        per_prompt = {}
        for r in mine:
            per_prompt.setdefault(r['index'], []).append(r['score'])
        assert len(per_prompt) == recipe.EVAL_EXAMPLES[domain] and len(mine) == len(per_prompt) * args.samples
        lengths = [r['response_tokens'] for r in mine]
        summary['domains'][domain] = {
            f'avg@{args.samples}': statistics.fmean(r['score'] for r in mine),
            f'pass@{args.samples}': statistics.fmean(max(s) for s in per_prompt.values()),
            'finished_thinking': statistics.fmean('</think>' in r['text'] for r in mine),
            'truncated': statistics.fmean(r['finish_reason'] == 'length' for r in mine),
            'response_tokens_mean': statistics.fmean(lengths), 'response_tokens_median': statistics.median(lengths)}
    with gzip.open(args.output / 'responses.jsonl.gz', 'wt') as stream:
        for response in responses:
            stream.write(json.dumps(response) + '\n')
    for path in args.output.glob('worker-*.jsonl.gz'):
        path.unlink()
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', type=Path, help='Hugging Face checkpoint directory')
    parser.add_argument('output', type=Path, help='new directory for the results')
    parser.add_argument('--domains', nargs='+', default=list(recipe.ALL_DOMAINS), choices=recipe.ALL_DOMAINS)
    parser.add_argument('--samples', type=int, default=8, help='responses per prompt (default: 8)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpus', type=int, default=8, help='one vLLM engine per GPU (default: 8)')
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        return worker(args)
    args.output.mkdir(parents=True)
    workers = [subprocess.Popen([sys.executable, __file__, *sys.argv[1:], '--worker', str(i)],
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(i)},
                                stdout=(args.output / f'worker-{i}.log').open('wb'), stderr=subprocess.STDOUT)
               for i in range(args.gpus)]
    codes = [w.wait() for w in workers]
    if any(codes):
        raise SystemExit(f'Workers exited with {codes}; see {args.output}/worker-*.log')
    summarize(args)


if __name__ == '__main__':
    main()
