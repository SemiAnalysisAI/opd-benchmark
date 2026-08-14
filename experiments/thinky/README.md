# Hosted OPD with Thinking Machines Tinker

This experiment runs the reverse-text on-policy distillation workload through
Thinking Machines' hosted Tinker service and compares it with the existing
self-hosted Prime-RL, Miles, and verl results.

It is based on Tinker's first-party
[on-policy distillation recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)
at commit `f46eddde86e5397138917516a6c69d2ecbf538b1`.

## Matched workload

- Dataset: `PrimeIntellect/Reverse-Text-RL`, preserving source order
- System prompt and LCS-ratio task score: identical to the local OPD benchmark
- 15 optimizer steps
- 8 prompts per step × 16 rollouts = 128 trajectories per step
- 128 maximum response tokens, temperature 1.0
- Sampled-token reverse KL with coefficient 1.0
- Zero task reward; task quality is recorded without affecting training
- Learning rate 3e-6

## Necessary provider differences

Tinker's current hosted catalog does not contain the local benchmark's
`PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT` student or
`PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL` teacher. The default hosted pair is
therefore `Qwen/Qwen3.5-4B` as student and `Qwen/Qwen3.5-9B` as teacher, using
the shared `qwen3_5_disable_thinking` renderer. The script verifies tokenizer
compatibility before starting paid training.

Tinker performs hosted LoRA training (rank 32 here), whereas the local runs
update the full 0.6B model. Treat the result as a provider/system comparison,
not a controlled model-quality ranking. Current supported models and token
prices are published in Tinker's
[machine-readable model catalog](https://tinker-docs.thinkingmachines.ai/tinker/models.json).

## Setup and run

```bash
cd experiments/thinky
uv sync

# Free preflight: dataset, tokenizer compatibility, renderer, and pricing.
uv run -- python scripts/preflight.py

export TINKER_API_KEY='...'

# Cheap API/recipe smoke test first.
STEPS=1 PROMPTS_PER_STEP=1 ROLLOUTS_PER_PROMPT=2 \
  SAVE_EVERY=1 uv run -- bash scripts/run.sh

# Matched 15-step workload.
uv run -- bash scripts/run.sh
```

Do not commit or paste the API key into chat or command arguments. If a key is
ever exposed, revoke it and export a replacement only in your local shell. The
run writes Tinker-native `metrics.jsonl`, config,
checkpoints, console output, wall time, `hosted_comparison.json`, and a compact
Markdown table under `results/<run-id>/`.

Runtime parameters can be overridden with environment variables such as
`STUDENT_MODEL`, `TEACHER_MODEL`, `STUDENT_CHECKPOINT`, `TEACHER_CHECKPOINT`,
`STEPS`, `PROMPTS_PER_STEP`, `ROLLOUTS_PER_PROMPT`, `MAX_TOKENS`,
`MAX_PROMPT_TOKENS`, `LEARNING_RATE`, `LORA_RANK`, and `RUN_ID`. Extra CLI flags are forwarded to
`run_opd.py`.

The cost shown by `analyze.py` is a lower/upper estimate using cached versus
uncached prefill prices. For account-grade token usage, use Tinker's documented
[`tinker billing usage`](https://tinker-docs.thinkingmachines.ai/tinker/cli/billing/)
command after its billing data has settled.

## Offline checks

```bash
python3 -m unittest discover -s experiments/thinky/tests -v
bash -n experiments/thinky/scripts/run.sh
```
