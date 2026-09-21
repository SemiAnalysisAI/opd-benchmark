# Tinker MOPD results — 2026-09-21

Two specialist teachers were trained remotely, then frozen for a fresh-base
Qwen/Qwen3.6-35B-A3B student. The student completed **40 optimizer updates,
128 responses/update, 5,120 training responses**, with 64 prompts from each
domain per update. All reported scores use the original 512 development
examples/domain, greedy decoding, thinking disabled, and a 256-token cap.
These are development results from one seed, not held-out test results.

## Measured quality

| Model/checkpoint | Countdown | Graph Coloring |
|---|---:|---:|
| Selected specialist (own domain), update 80 | 238/512 (46.48%) | 466/512 (91.02%) |
| Fresh student, update 0 | 59/512 (11.52%) | 130/512 (25.39%) |
| MOPD student, update 10 | 170/512 (33.20%) | 277/512 (54.10%) |
| MOPD student, update 20 | 202/512 (39.45%) | 317/512 (61.91%) |
| MOPD student, update 30 | 214/512 (41.80%) | 413/512 (80.66%) |
| MOPD student, update 40 | **215/512 (41.99%)** | **436/512 (85.16%)** |

The specialist row represents two different teachers. Each received 80 updates
of 32 prompts × 8 responses; the first 40-update checkpoints are also retained.
The graph teacher met its provisional 90% goal. Countdown missed its provisional
50% goal and used its best scheduled checkpoint at the 80-update cap.
**Original teacher scores were not available, so teacher-quality matching is
not established.** Those provisional goals are not recovered original scores.
Selection used development scores only; the student always uses update 40.

Final student evaluation had 49/512 Countdown responses reach the token cap
and zero graph responses reach it. All responses, including truncations,
were scored and retained without filtering.

## Table for the service comparison

| Execution | Countdown baseline → final | Graph baseline → final | Observed student duration |
|---|---:|---:|---:|
| Miles, self-hosted | 10.55% → 45.51% | 26.17% → 88.28% | 21m 34s allocation |
| Prime-RL, self-hosted | 10.94% → 39.65% | 25.78% → 64.65% | 24m 22s allocation |
| Slime, self-hosted | 10.94% → 44.14% | 26.37% → 89.26% | 13m 25s allocation |
| Tinker, hosted | 11.52% → 41.99% | 25.39% → 85.16% | 31m 31s client elapsed |

Original observations and timing definitions come from
[the repository provenance](../../docs/PROVENANCE.md). All students have the
same 40-update/5,120-response budget, but this is not a controlled ranking.
The self-hosted runs used 16 B200 GPUs; Tinker hardware and exact served weight
revision are provider-controlled. Tinker used rank-64 LoRA, LR 1e-5 (the
cookbook's 10× conversion from the original full-finetuning LR 1e-6), synchronous
sampling, and sampled-token reverse-KL distillation with no task reward.
Teacher identities, objectives, scheduling, and stopping rules differ.
Tinker stops at EOS; the pinned Miles launcher defaults to an answer-tag stop.

Tinker student elapsed time was 1,891.39 seconds, including 418.43 seconds of
evaluation. Update loops totaled 1,397.77 seconds (median 34.97 seconds/update).
This excludes specialist preparation and earlier attempts. The recorded
campaign interval including attempts, gaps, and overlapping teachers was
96.02 minutes; it excludes research/setup before the first recorded attempt.

## Evidence and reproducibility

Full artifacts are retained outside Git at
`/Users/joey/workspace/ssb/opd-runs/tinker-20260921/`:

- `REPORT.md`, `summary.json`, and `learning-curves.png/.svg`: all attempts,
  evaluations, timings, and cost qualifications.
- `JOURNAL.md`: decisions, errors, fixes, and teacher-selection amendments.
- Per-run source/config/package/data snapshots, console logs, `samples.jsonl`,
  `learner-scores.jsonl`, `teacher-scores.jsonl`, `metrics.jsonl`, and evaluations.
- `teacher-selection.json`, `teachers.json`, and per-run `checkpoints.jsonl`:
  exact private remote model and optimizer-state paths.
- `final-audit.json`: all 40 updates, 5,120 training sequences, and 5,120
  evaluation sequences verified; every task score recomputed; every routed
  distillation advantage checked. Seven Tinker tests and six package tests pass.
- `service-metadata-20260921T104053Z/`: run/checkpoint metadata and billing snapshot.

The final student sampler is
`tinker://e7a10aea-05ff-554d-bbe6-a1ba0511c480:train:0/sampler_weights/step-040`.
The optimizer state uses the same run prefix with `weights/step-040`.
No model weights were downloaded; checkpoints remain private on Tinker.

The main execution issue was explicit seeding of multi-response teacher calls,
which produced repeated responses within groups. Both initial teacher attempts
were invalidated and retained; corrected teachers restarted from fresh base
with the multi-response seed unset, following the cookbook. Other fixes were
explicit tokenizer `return_dict=False`, a floating-point test tolerance, and
disabling scorer-import bytecode in the original campaign template directory.
The corrected teachers, extensions, and student all exited successfully.

Billing had no settled usage rows at collection time; this does not mean zero
cost. Recorded-token scenarios are $2.57/$3.74 for the student and $20.77/$25.86
across recorded attempts with all-cached/all-uncached prefill. They are not
invoices or total-cost bounds and exclude unlogged retries, unfinished mutations,
probes, and storage. Forty retained remote checkpoints total 357.67 decimal GB,
approximately $35.77/month at the documented storage rate if kept for a full
month, subject to billing units and retention time.

See [the protocol and official documentation links](README.md) for the complete
recipe and [run.py](run.py) for execution. Original self-hosted recipes were not
changed by this campaign.
