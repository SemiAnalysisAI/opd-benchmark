# OPD and MOPD on a hosted training API

Two separate experiments with one shared loop:

| Script | Experiment |
|---|---|
| `opd.py` | **On-policy distillation.** A fresh student learns one domain from one frozen teacher. |
| `mopd.py` | **Multi-teacher OPD.** A fresh student learns several domains at once, each response scored by its own domain's teacher. |
| `common.py` | The shared pieces: prompts, rollouts, the distillation update, evaluation, records. |
| `backend.py` | The service. Only the Tinker client API is used, so a Tinker-compatible provider is one more class. |
| `report.py` | Tables, figures, throughput, and a cost estimate from the saved records. |
| `test_opd.py` | Offline tests. |

## What one update does

1. The student samples one response per training prompt (temperature 1, thinking on, up to 30,720 tokens).
2. The prompt's teacher scores every response token: a prefill, no generation. Scoring starts as soon as a
   response finishes, while the others are still generating.
3. The per-token advantage is **teacher logprob − student logprob**, the negative sampled-token reverse KL.
   There is no task reward.
4. One importance-sampling policy-gradient step on those advantages (`loss_fn='importance_sampling'`).
5. The new weights are saved and published to the sampler, so the next update samples from them.

By default the loop is asynchronous, as in `shared/recipe.py` (`MAX_POLICY_LAG = 1`) and `hosted/tinker/run.py`'s
`--pipeline-depth 1`: once update n's responses are in, update n+1's prompts are sampled and scored on the same
weights while update n trains and publishes. Every batch is at most one optimizer step stale; each rollout and
metrics row records its `policy_lag`, and the loss uses the logprobs of the weights that sampled it.
`--policy-lag 0` is the synchronous variant (every response from the current weights), as in the cookbook's
`tinker_cookbook/distillation/train_on_policy.py`. The Tinker and Fireworks runs of 2026-09-27/28 used
`--policy-lag 0`, which was then the only mode.

## Protocol

The defaults are `shared/recipe.py`'s, so the runs compare with the self-hosted frameworks:

- Student: fresh rank-64 LoRA on `Qwen/Qwen3.6-35B-A3B` (tokenizer pinned at
  `995ad96eacd98c81ed38be0c5b274b04031597b0`), LR 1e-5 (the recipe's 1e-6 × the cookbook's 10× for LoRA),
  Adam betas 0.9/0.999, gradient clip 1, no weight decay.
- 20 updates × 128 prompts × 1 response. OPD takes all 128 from its domain; MOPD takes 64 from each.
  Each domain's training split is shuffled once with a fixed seed, so OPD and MOPD see the same caesar
  prompts in the same order (OPD just gets through them twice as fast).
- Evaluation at updates 0, 10 and 20, on **every** domain (so OPD also shows the domain it does not train
  on): the first 100 dev problems, 3 samples each at temperature 1. This is the teacher benchmark of
  `hosted/tinker/sft.py`, so student, teacher and base-model numbers are directly comparable.
- Teachers: the SFT teachers from `hosted/tinker/launch.py --teachers sft` (see `hosted/tinker/README.md`).

## Run

Python 3.12 with `hosted/tinker/requirements.txt`, `reasoning-gym==0.1.25` (or its site-packages in
`$CAMPAIGN_PYDEPS`), the pinned tokenizer cached, and `TINKER_API_KEY` exported (or a
`config/tinker.local.json`, which can also pin the account).

```bash
TEACHERS=/abs/sft-teachers/teachers.json   # {domain: sampler path}

python hosted/opd/opd.py --domain caesar_cipher --output /abs/opd-caesar \
    --teacher "$(python -c "import json; print(json.load(open('$TEACHERS'))['caesar_cipher'])")"
python hosted/opd/mopd.py --teachers $TEACHERS --output /abs/mopd

python hosted/opd/report.py /abs/opd-caesar /abs/mopd --output /abs/report --teachers-campaign /abs/sft-teachers
python -B -m unittest discover -s hosted/opd -p 'test_*.py'
```

Every flag is listed by `--help` (updates, prompts, LR, rank, evaluation size, concurrency). A run resumes
from an optimizer-state checkpoint into a new directory with `--resume STATE --start-update N`.

## What a run saves

| File | Contents |
|---|---|
| `config.json`, `teachers.json`, `versions.json`, `source/`, `data-manifest.json` | Exact inputs: settings, teacher paths, package versions, a copy of the code, data hashes |
| `account.json`, `capabilities.json`, `model-info.json` | The verified account and the service's view of the model and LoRA |
| `rollouts/update-NNN.jsonl.gz` | Every training response: prompt and response tokens, student and teacher logprobs, advantages, text, task score, stop reason, and when it was submitted, sampled and scored |
| `rollouts/eval-NNN.jsonl.gz` | Every evaluation response with its tokens, text and score |
| `metrics.jsonl` | Per update: tokens, truncations, per-domain reverse KL and score, timings, throughput, trainer-versus-sampler logprob gap, sampler path, service metrics |
| `evaluations.jsonl` | Per evaluation: avg@3, pass@3, mean tokens and truncations per domain |
| `checkpoints.jsonl` | Optimizer-state paths at each evaluation (for resuming or further training) |
| `events.jsonl` | Everything that happened, with UTC and elapsed time; a failure also writes `error.txt` |

Weights stay on the provider; only paths are recorded.

## Timing definitions

All times are client wall time on a shared hosted service, not GPU utilization.

- **rollout**: from submitting the update's prompts until every response is sampled *and* teacher-scored.
- **median / p90 / last response**: when those responses finished sampling, from the update's start. The
  gap between median and last is the straggler cost of long thinking responses in a synchronous update.
- **teacher tail**: scoring still running after the last response finished.
- **train**: `forward_backward` plus `optim_step`. **sync**: saving weights and pointing the sampler at them.
- **sampled tokens/s**: the update's response tokens over the time to its last response.
