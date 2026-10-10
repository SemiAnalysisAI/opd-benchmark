# Tinker: hosted multi-teacher OPD

This runs the shared MOPD recipe (`shared/recipe.py`: reasoning_gym `caesar_cipher`
and `simple_geometry`, thinking on) on the hosted Tinker service. It trains one
specialist teacher per domain, then distills a fresh student from both with routed
sampled-token OPD. Unlike the self-hosted frameworks, which distill the frozen
`recipe.TEACHERS`, this campaign builds its own teachers, by GRPO (default) or by SFT.
Model and optimizer checkpoints stay on Tinker; only the tokenizer, source and logs
are local.

No student results exist yet for this campaign. [RESULTS.md](RESULTS.md) holds the
previous Countdown/Graph Coloring experiment (thinking off). Tinker OPD and MOPD runs
with the `hosted/opd` loop are in [../opd/RESULTS.md](../opd/RESULTS.md).

| File | Purpose |
|---|---|
| `run.py` | One run: `eval`, `teacher` (GRPO) or `mopd` (student). |
| `launch.py` | The full campaign: account check, baseline, teachers, selection, student, report. |
| `report.py` | Audits the student and writes `final-audit.json`, `summary.json`, `REPORT.md`, `learning-curves.png`. |
| `test_run.py` | Offline tests: token alignment, data, prompt template, scoring, selection, launcher settings, pipelining. |
| [`../opd/sft.py`](../opd/sft.py) | SFT teachers (`train`) and the 100 × 3 teacher benchmark (`bench`); `launch.py --teachers sft` runs it. Its tests are in `../opd/test_sft.py`. |

## Run

Requirements:

- Python 3.12 with `requirements.txt` installed (`matplotlib` is optional, for the plot).
- `reasoning-gym==0.1.25` installed, or its site-packages directory in `$CAMPAIGN_PYDEPS`
  (see `shared/scoring.py`).
- The pinned tokenizer already in the Hugging Face cache; the launcher sets `HF_HUB_OFFLINE=1`.
- `TINKER_API_KEY` exported. To read the key from `~/.zprofile` instead, or to pin an
  account, copy `config/tinker.example.json` to `config/tinker.local.json` and set
  `api_key_profile_label` or `expected_email`/`expected_org`. With the guard set, the
  launcher refuses any other account. See [config/README.md](../../config/README.md).

```bash
# Full campaign into a new directory.
python hosted/tinker/launch.py --root /abs/new-campaign

# Student only, reusing a finished campaign's selected teachers.
python hosted/tinker/launch.py --root /abs/new-student --teacher-campaign /abs/new-campaign

# SFT teachers and their benchmark only (optionally --sft-config FILE), then a student from them.
python hosted/tinker/launch.py --root /abs/sft-teachers --teachers sft --teachers-only
python hosted/tinker/launch.py --root /abs/sft-student --teacher-campaign /abs/sft-teachers

# One run directly.
python hosted/tinker/run.py mopd --teachers teachers.json --output /abs/student

# Offline tests.
python -B -m unittest discover -s hosted/tinker -p 'test_*.py' -v
```

The launcher copies `hosted/tinker`, `hosted/opd`, `shared` and `data/` into
`<root>/source-repo` and runs every stage from that snapshot. Progress is in
`status.json`, `campaign-events.jsonl` and `<stage>-console.log`.

Each run directory keeps its config, package versions, source, scorer, data hashes,
every sample's tokens, logprobs and text, teacher and learner scores, metrics,
evaluations, and remote checkpoint paths.

## Protocol

- Base model `Qwen/Qwen3.6-35B-A3B`. The tokenizer is pinned at
  `995ad96eacd98c81ed38be0c5b274b04031597b0`; the hosted weights are provider-controlled.
  Prompts use the cookbook `qwen3_5` renderer, and its tokens (ending in the `<think>`
  prefill) must equal the pinned HF chat template's with `enable_thinking=True`.
- The exact `data/` train/dev splits and the shared scorer. Seed 20260921, with an
  independent deterministic shuffle per domain.
- Rank-64 LoRA (attention, MLP, unembedding), LR `1e-5`. This is the cookbook's
  documented 10× FullFT-to-LoRA conversion of the recipe's `1e-6`, which is also the
  LR of the released GRPO teachers. It is a heuristic, not an equivalence.
- GRPO teachers use the training settings of the released GRPO teachers: one
  independently initialized LoRA per domain; 32 prompts × 8 responses, temperature 1,
  top-p 1; PPO clip 0.8/1.2; per-group mean/std reward normalization; Adam betas
  0.9/0.98, no weight decay, gradient clip 1. Each teacher is evaluated every 25 updates
  and stops at the first evaluation that meets its goal (caesar_cipher 70%,
  simple_geometry 95%), or after 200 updates. Selection takes the first checkpoint that
  meets the goal, otherwise the best (earliest on ties). Two differences from the
  released teachers' training: groups with no reward variance are kept (zero advantage),
  and training is synchronous.
- Student: a fresh base model, 20 updates of 64 prompts per domain × 1 response (2,560
  trajectories). Sampled-token reverse KL (teacher minus sampler logprob) with native
  importance sampling and zero task reward; Adam betas 0.9/0.999, no weight decay,
  gradient clip 1. Losses sum over response tokens, so equal prompt counts do not mean
  equal per-domain gradient weight.
- Up to 30,720 response tokens in a 32,768-token context. Generation stops only at the
  model's end-of-turn token. A response scores 1 only if the text after its last
  `</think>` contains an `<answer>` that reasoning_gym accepts.
- Evaluation at updates 0, 10 and 20: the full dev split (126 caesar_cipher, 128
  simple_geometry), greedy.
- Multi-response teacher requests leave the sampling seed unset, as in the cookbook.
  Rollouts are reproduced from the retained token records, not by regeneration.
- Optimizer calls are never retried by the application. To resume, start from a
  recorded state checkpoint in a fresh directory.

## Pipelining

The launcher runs the student with `--pipeline-depth 1` (pass 0 for synchronous
training). It prepares the next batch, with teacher scoring overlapping generation,
while the current update trains. Policy lag is at most one update
(`recipe.MAX_POLICY_LAG`) and is audited; the budget stays exactly 20 × 128 responses.

The advantage is still teacher minus behavior logprob under importance sampling, so
this is an off-policy variant of the synchronous objective, and its quality should be
measured. In pipelined runs, `sampling_seconds` is the exposed wait for the prepared
batch.

## SFT teachers

`hosted/opd/sft.py` builds a teacher by supervised fine-tuning instead of GRPO, and
benchmarks it. Its defaults reproduce the released SFT teachers below. Every setting is
a flag and is recorded in the run's `config.json` (`changed_from_defaults` lists what
differs).

- Data: by default every trace of
  [`semianalysisai/Qwen3.6-35B-A3B-teacher-sft-caesar-geometry`](https://huggingface.co/datasets/semianalysisai/Qwen3.6-35B-A3B-teacher-sft-caesar-geometry)
  at `b21bb3e75b1ca903f4b69bd609adc89bd99543f1`: verified-correct traces (temperature 1,
  thinking on) sampled from the frozen `recipe.TEACHERS`, 16,980 caesar_cipher and
  29,547 simple_geometry rows. No dev problem appears in it. Every row is checked
  (roles, system prompt, one `</think>`) before training.
- Rendering: the pinned HF template. Each prompt must equal the cookbook renderer's
  generation prompt and be a prefix of the rendered trace. The loss covers only the
  completion: thinking, answer and `<|im_end|>`.
- Training: one rank-64 LoRA per domain, on that domain's rows only. Summed token cross
  entropy, one epoch, 128 sequences per update, final partial batch dropped (132 and 230
  updates), seeded shuffle. LR is the cookbook's `get_lr` for this model (about 5e-4)
  with linear decay to zero; Adam betas 0.9/0.95, no weight decay or clipping. Optimizer
  state is saved every 25 updates, sampler weights at the end.
- Benchmark: the first 100 problems of each dev split, 3 samples each at temperature 1,
  top-p 1, up to 30,720 response tokens. `benchmark.json` reports avg@3 (`accuracy`),
  pass@3, all-3-correct, lengths, truncations and unfinished thinking. Each teacher is
  benchmarked on its own domain when training ends.

| Flag | Default | Meaning |
|---|---|---|
| `--dataset`, `--revision` | the dataset above, pinned | Any dataset with the same layout (see the `sft.py` docstring) |
| `--min-correct` | 1 | Keep prompts the teacher solved at least this many times of 3 (3 = easiest only) |
| `--traces-per-prompt` | 0 (all) | At most this many traces per prompt, lowest sample index first |
| `--max-rows` | 0 (all) | Then a seeded random subset of this many rows |
| `--learning-rate`, `--lr-schedule` | `get_lr` (≈5e-4), `linear` | Peak LR; `linear`, `cosine` or `constant` |
| `--batch-size`, `--epochs`, `--rank` | 128, 1, 64 | Each epoch reshuffles; epoch 0 is the released order |
| `--save-every` | 25 | Optimizer-state checkpoint interval, in updates |
| `--bench-examples`, `--bench-samples`, `--bench-temperature` | 100, 3, 1.0 | Benchmark protocol |

```bash
# Reproduce one released teacher, or benchmark a model (the base model by default).
python hosted/opd/sft.py train --domain caesar_cipher --output /abs/teacher-caesar-cipher
python hosted/opd/sft.py bench --output /abs/bench [--checkpoint tinker://.../sampler_weights/final]

# A different teacher: one trace per prompt, only prompts the teacher always solved, two epochs.
python hosted/opd/sft.py train --domain caesar_cipher --output /abs/t \
    --traces-per-prompt 1 --min-correct 3 --epochs 2

# Resume from an optimizer-state checkpoint (see checkpoints.jsonl) into a fresh directory, same settings.
python hosted/opd/sft.py train --domain caesar_cipher --output /abs/t-resumed \
    --checkpoint tinker://.../weights/step-0100 --start-step 100
```

`launch.py --teachers sft` runs the base-model benchmark alongside both teachers, then
writes `teachers.json` and `teacher-selection.json` so that `--teacher-campaign` can
reuse the teachers for a student. There is no selection: each teacher is its final
checkpoint. `--sft-config FILE` passes `sft.py` settings, for every stage or per domain,
and is saved as `sft-config.json` in the campaign:

```json
{"learning_rate": 2e-4, "domains": {"caesar_cipher": {"epochs": 2}, "simple_geometry": {"max_rows": 5000}}}
```

### Released SFT teachers (2026-09-27)

Built with `launch.py --teachers sft --teachers-only` and the defaults, on the SA RL
Infra Research account (campaign `tinker-sa-sft-teachers-20260927T115503Z`). The two
teachers trained concurrently, about 20 minutes each.

| | caesar_cipher base | caesar_cipher SFT teacher | simple_geometry base | simple_geometry SFT teacher |
|---|---|---|---|---|
| avg@3 | 27.3% | 79.3% | 51.3% | 99.7% |
| pass@3 | 47% | 95% | 81% | 100% |
| all 3 correct | 12% | 57% | 19% | 99% |
| mean response tokens | 11.1k | 12.7k | 3.6k | 3.1k |
| truncated (of 300) | 13 | 13 | 1 | 0 |

- Sampler paths: caesar_cipher
  `tinker://02c4683b-edf3-526b-991d-90c9a0c1b183:train:0/sampler_weights/final`,
  simple_geometry
  `tinker://231067a0-323a-5424-b0f2-62353750d220:train:0/sampler_weights/final`.
  They stay on the account that trained them.
- Training tokens: 199.0M and 93.5M, about $345 at the published $1.177/M, plus a few
  dollars of sampling. This is a list-price estimate, not an invoice.
- Train NLL barely moves (0.316 → 0.308 and 0.342 → 0.331, first vs last 20
  updates). The traces come from a GRPO-trained copy of the same base, so little
  per-token loss is reducible; accuracy is what changes.
- For comparison, the GRPO teachers score 73.2% and 99.5% avg@8 on the full dev splits.
  The base scores here match the dataset card's (27.8% and 52.0%), but 100 problems × 3
  samples is a smaller sample.

## Caveats

- Thinking at 32k tokens costs far more than the previous 256-token protocol. Responses
  in the released GRPO teachers' training averaged about 11k (caesar_cipher) and 4k
  (simple_geometry) tokens, and a GRPO teacher can run 200 updates of 256 responses.
  Watch the cost estimate in `REPORT.md` while the campaign runs.
- The GRPO teacher goals are operational, set just below the released GRPO teachers'
  held-out scores (27.0% → 73.3% and 53.9% → 99.8%, sampled at temperature 1). Meeting
  them does not show that the retrained teachers match `recipe.TEACHERS`.
- Equal steps and prompts do not make this a controlled comparison with the self-hosted
  frameworks: Tinker uses LoRA, hosted scheduling and hardware, sampled-token OPD, and
  its own teachers. Timings are client wall time, not GPU utilization.
- Report teacher training cost separately from the student, including failed attempts.

## Sources

- [Miles PR 3116](https://github.com/radixark/miles/pull/3116) at
  `8f8e4dff55d80bd9d36c3490c333a119f38480ed`.
- Tinker docs: [distillation](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/),
  [LoRA](https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/),
  [PPO](https://tinker-docs.thinkingmachines.ai/tinker/losses/ppo/),
  [importance sampling](https://tinker-docs.thinkingmachines.ai/tinker/losses/importance-sampling/).
- Tinker cookbook `recipes/rl_loop.py`, `distillation/train_on_policy.py` and
  `hyperparam_utils.py` at `1e53aa3d1cdd6389b3290c2574641eccc0503242` (runtime 0.5.7).
