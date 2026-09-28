# Tinker MOPD results, 2026-09-21 (previous experiment)

These results are from the earlier Countdown/Graph Coloring campaign: thinking off,
256 response tokens, 40 student updates, 512 dev puzzles per domain. Its code is in git
history (commit `3e27ddb`, `frameworks/tinker/`). The current code runs the
reasoning_gym experiment in `shared/recipe.py`, which has no recorded results yet.

Two specialist teachers were trained on Tinker, then frozen and used to distill a fresh
`Qwen/Qwen3.6-35B-A3B` student. The student ran 40 optimizer updates of 128 responses
(64 prompts per domain), 5,120 training responses in total. All scores use the 512
development examples per domain, greedy decoding, thinking off, and a 256-token cap.
They are development results from one seed, not held-out test results.

## Quality

| Model/checkpoint | Countdown | Graph Coloring |
|---|---:|---:|
| Selected specialist (own domain), update 80 | 238/512 (46.48%) | 466/512 (91.02%) |
| Fresh student, update 0 | 59/512 (11.52%) | 130/512 (25.39%) |
| MOPD student, update 10 | 170/512 (33.20%) | 277/512 (54.10%) |
| MOPD student, update 20 | 202/512 (39.45%) | 317/512 (61.91%) |
| MOPD student, update 30 | 214/512 (41.80%) | 413/512 (80.66%) |
| MOPD student, update 40 | 215/512 (41.99%) | 436/512 (85.16%) |

The specialist row covers two teachers. Each ran 80 updates of 32 prompts × 8
responses; the 40-update checkpoints are also retained. The graph teacher met its
provisional 90% goal. The Countdown teacher missed its provisional 50% goal, so its best
scheduled checkpoint at the 80-update cap was used. The original teacher scores were
not available, so the goals are not recovered scores and teacher quality is not shown to
match the originals. Selection used development scores only; the student always uses
update 40.

In the final student evaluation, 49/512 Countdown responses and no Graph Coloring
responses hit the token cap. All responses, including truncated ones, were scored and
kept.

## Comparison with the self-hosted runs

| Execution | Countdown baseline → final | Graph baseline → final | Observed student duration |
|---|---:|---:|---:|
| Miles, self-hosted | 10.55% → 45.51% | 26.17% → 88.28% | 21m 34s allocation |
| Prime-RL, self-hosted | 10.94% → 39.65% | 25.78% → 64.65% | 24m 22s allocation |
| Slime, self-hosted | 10.94% → 44.14% | 26.37% → 89.26% | 13m 25s allocation |
| Tinker, hosted | 11.52% → 41.99% | 25.39% → 85.16% | 31m 31s client elapsed |

The self-hosted numbers and timing definitions come from
[PROVENANCE.md](../../docs/PROVENANCE.md). Every student had the same budget of 40
updates and 5,120 responses, but this is not a controlled ranking:

- The self-hosted runs used 16 B200 GPUs. Tinker's hardware and exact served weight
  revision are provider-controlled.
- Tinker used rank-64 LoRA, LR 1e-5 (the cookbook's 10× conversion of the original
  full-finetuning LR 1e-6), synchronous sampling, and sampled-token reverse-KL
  distillation with no task reward.
- Teacher identities, objectives, scheduling and stopping rules differ.
- Tinker stops at EOS; the pinned Miles launcher defaults to an answer-tag stop.

## Timing

The Tinker student took 1,891.39 seconds, including 418.43 seconds of evaluation. The
update loops took 1,397.77 seconds (median 34.97 seconds per update). This excludes
teacher training and earlier attempts. The whole recorded campaign, including attempts,
gaps and overlapping teachers, took 96.02 minutes; it excludes setup before the first
recorded attempt.

## Cost

Billing had no settled usage rows when collected, which does not mean zero cost.
Estimates from recorded tokens are $2.57 (all prefill cached) to $3.74 (all uncached)
for the student, and $20.77 to $25.86 across all recorded attempts. These are not
invoices or cost bounds: they exclude unlogged retries, unfinished mutations, probes and
storage. The 40 retained remote checkpoints total 357.67 GB (decimal), about
$35.77/month at the documented storage rate if kept for a full month, depending on
billing units and retention time.

## Issues found during the run

The main problem was explicit seeding of multi-response teacher calls, which produced
repeated responses within groups. Both initial teacher attempts were invalidated (and
kept); the corrected teachers restarted from the base model with the multi-response
seed unset, as in the cookbook. Smaller fixes: explicit tokenizer `return_dict=False`, a
floating-point test tolerance, and disabling bytecode for scorer imports in the original
campaign template directory. The corrected teachers, their extensions and the student
all exited successfully.

## Artifacts

Full artifacts are kept outside Git in a private run archive, `opd-runs/tinker-20260921/`:

- `REPORT.md`, `summary.json`, `learning-curves.png/.svg`: all attempts, evaluations,
  timings and cost notes.
- `JOURNAL.md`: decisions, errors, fixes and teacher-selection amendments.
- Per-run source, config, package and data snapshots, console logs, `samples.jsonl`,
  `learner-scores.jsonl`, `teacher-scores.jsonl`, `metrics.jsonl` and evaluations.
- `teacher-selection.json`, `teachers.json` and per-run `checkpoints.jsonl`: the private
  remote model and optimizer-state paths.
- `final-audit.json`: all 40 updates, 5,120 training sequences and 5,120 evaluation
  sequences verified, every task score recomputed, every routed distillation advantage
  checked. Seven Tinker tests and six package tests passed.
- `service-metadata-20260921T104053Z/`: run and checkpoint metadata and a billing
  snapshot.

The final student sampler is
`tinker://e7a10aea-05ff-554d-bbe6-a1ba0511c480:train:0/sampler_weights/step-040`; the
optimizer state has the same run prefix with `weights/step-040`. No weights were
downloaded; the checkpoints remain private on Tinker. The self-hosted recipes were not
changed by this campaign.
