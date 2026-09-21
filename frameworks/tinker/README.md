# Tinker puzzle teachers and MOPD

Remote Qwen3.6-35B-A3B training using Tinker 0.30.0 and cookbook 0.5.7.
Only tokenizers, source, and logs are stored locally. Model and optimizer
checkpoints stay on Tinker. This directory is a standalone SDK campaign;
the existing Slurm preparation/submission tools do not launch it.

The completed 2026-09-21 campaign is documented in [RESULTS.md](RESULTS.md),
including the 40-update student result and comparison qualifications.

## Protocol fixed before training

- Exact `data/` train/dev files and original standard-library verifier.
- Pinned base tokenizer revision `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- Hosted base model `Qwen/Qwen3.6-35B-A3B`; hosted weight revision is provider-controlled.
- Thinking disabled; pinned HF prompt tokens must equal cookbook renderer tokens.
- Two independently initialized domain teachers: 40 GRPO updates, 32 prompts
  × 8 responses, 256 response tokens, temperature 1, top-p 1.
- Rank-64 LoRA on attention, MLP, and unembedding; teacher LR `1e-5`.
- This LR applies the cookbook's documented 10× FullFT-to-LoRA conversion to
  the original `1e-6`; it is a heuristic, not an equivalence guarantee.
- Teacher PPO clips at 0.8/1.2; per-group mean/std reward normalization.
  Adam betas 0.9/0.98, weight decay 0.1, gradient clipping 1.
- Student: fresh base initialization, 40 optimizer updates, exactly 64 prompts
  per domain per step and one response each (5,120 total trajectories), 256
  response tokens, sampled-token reverse KL coefficient 1, zero task reward.
- Student default LR `1e-5` uses the same documented LoRA conversion. A literal
  `1e-6` control can be run separately and must be labeled separately.
- Student Adam betas 0.9/0.999, no weight decay, gradient clipping 1.
- Native Tinker losses sum response-token contributions; equal prompt counts
  do not imply equal token counts or equal per-domain gradient weight. No
  additional domain-loss renormalization is applied.
- Baseline and every 10 updates: fixed weights, all 512 dev puzzles per domain,
  greedy decoding, same response cap and scorer. Test data is not used for tuning.
- Seed 20260921, independent deterministic domain shuffles; synchronous weight
  publication after each update means zero accepted policy lag.
- Multi-sample teacher requests leave the sampling seed unset, as in the
  cookbook: explicitly setting one seed repeated responses within the group in
  the first attempts. Initialization/order are seeded; full rollout replay uses
  the retained token records rather than claiming deterministic regeneration.
- Preserve all attempts. No application-level automatic retry of optimizer
  mutations. Resume only from a recorded state checkpoint in a fresh directory.

The original pinned teacher cards report their training recipe but omit their
development accuracy. Teacher-quality matching therefore remains unverified
until the original evaluation records or explicit target scores are supplied.
Do not substitute the PR's student results for teacher targets.

## Sources

- [Miles PR 3116](https://github.com/radixark/miles/pull/3116), pinned revision
  `8f8e4dff55d80bd9d36c3490c333a119f38480ed`.
- [Tinker distillation documentation](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)
- [Tinker LoRA guidance](https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/)
- [Tinker PPO loss](https://tinker-docs.thinkingmachines.ai/tinker/losses/ppo/)
- [Tinker importance sampling loss](https://tinker-docs.thinkingmachines.ai/tinker/losses/importance-sampling/)
- Cookbook `recipes/rl_loop.py`, `distillation/train_on_policy.py`, and
  `hyperparam_utils.py`. Research clone revision:
  `1e53aa3d1cdd6389b3290c2574641eccc0503242`; installed runtime version: 0.5.7.

## Commands

Use Python 3.12 with `tinker==0.30.0`, `tinker-cookbook==0.5.7` installed.
The key is read from the environment; do not put it in command arguments.

```bash
python frameworks/tinker/run.py eval --output /absolute/run/baseline
python frameworks/tinker/run.py teacher --domain countdown --output /absolute/run/countdown
python frameworks/tinker/run.py teacher --domain graph_color --output /absolute/run/graph-color
python frameworks/tinker/run.py mopd --teachers /absolute/run/teachers.json --output /absolute/run/student
```

`teachers.json` maps `countdown` and `graph_color` to the selected persistent
`tinker://.../sampler_weights/...` paths in the teacher checkpoint records.

Raw artifacts belong outside the checkout. Each run saves source, resolved
arguments, package versions, dataset hashes, sample token IDs/logprobs/text,
teacher scores, optimizer metrics, stage timing, errors, evaluation summaries,
and remote checkpoint paths. Console output should also be redirected to a log.

### Full campaign with an account check

`launch_campaign.py` reads one literal key assignment from `~/.zprofile`, checks
its email and organization with Tinker, and refuses to train if either differs
from the expected values. It overrides the key only in its process and children;
it does not edit the profile or fall back to another account. The key is never
written to the campaign logs or command arguments.

```bash
python frameworks/tinker/launch_campaign.py \
  --root /absolute/new-campaign-directory \
  --credential-label SA_TINKER_API_KEY \
  --expected-email you@company.com \
  --expected-org "Your organization"
```

Use a fresh output directory. The controller snapshots code and data, evaluates
the base model, trains both teachers concurrently, extends unmet provisional
goals up to 80 updates, selects teachers, then trains the fresh 40-update student.
It runs the artifact audit, collects checkpoint/billing metadata, and generates
the report and plots. Install `matplotlib==3.11.2` in addition to the frozen
training requirements for plotting. The pinned tokenizer must already be cached:
this launcher sets `HF_HUB_OFFLINE=1`, and never downloads model weights.

`status.json`, `campaign-events.jsonl`, and individual console logs show progress
and failures. Keep the controller machine awake and connected for the entire
campaign. Tinker's SDK can pause on billing errors; inspect the logs if progress
stops. The controller does not automatically restart failed optimizer updates.

## Interpretation

Equal steps and prompt counts do not make this a controlled framework ranking:
Tinker uses LoRA, hosted scheduling/hardware, sampled-token OPD, and independent
retrained teachers. Miles uses candidate OPD; all three recorded self-hosted
campaigns differ in scheduling and numerical details. Report specialist-training
cost separately from student consolidation, include failed attempts, and compare
measured quality alongside wall time. No GPU-utilization claim can be made for
the hosted service without provider telemetry.

## Offline tests

```bash
python -m unittest discover -s frameworks/tinker -p 'test_*.py' -v
```

## Teacher extension decision — 2026-09-21T09:36:20.465387+00:00

After observing graph teacher dev 70.90% at update 30 (and before the 40-update results), adopt provisional specialist goals of 50% Countdown and 90% graph, with a maximum of 80 total updates per teacher in this extension. These are operational goals informed by the existing student task strengths, NOT recovered original teacher scores and NOT evidence of matched teacher distributions. Original teacher scores remain unavailable. Keep both initial 40-update results. If needed, resume exact optimizer state at update 40, preserve data ordering at the next batch, same LR/rank/objective, evaluate every 10, and stop a teacher at its first scheduled evaluation meeting its goal. Student stays fresh base, 40 updates, 128 prompts/update. Include all teacher preparation and failed-attempt costs separately from student cost.

### Stopping-rule qualification

Tinker uses model end-of-turn token 248046, without an `</answer>` string stop.
The pinned Miles launcher defaults to `stop_at_answer=True`; supplied Slime and
Prime-RL campaign configs do not specify that answer stop. Exact original teacher
command lines are not in the pinned model cards. All Tinker evaluations keep the
same EOS-only rule; post-answer continuations and truncation are retained.
