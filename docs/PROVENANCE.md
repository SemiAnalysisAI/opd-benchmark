# Provenance and interpretation

## Current experiment

The recipe in `shared/recipe.py` follows the research campaign `research/mopd-2026-09-25`: 20 updates, policy lag 1, thinking on, and 32k context.
That campaign had no in-loop evaluation. Its Miles run also used no context parallelism, after tuning.
**This repository's code has not run this experiment.**

Research campaign results (2026-09-25; separate campaign code, sampling at temperature 1, 8 samples; not produced by this repository's code).
Full dev sets: 126 `caesar_cipher` and 128 `simple_geometry` problems.

| Model | `caesar_cipher` avg@8 / pass@8 | `simple_geometry` avg@8 / pass@8 |
|---|---|---|
| Base | 27.8% / 69.8% | 52.0% / 98.4% |
| `caesar_cipher` teacher (step 125) | 73.2% / 100% | 57.1% / 99.2% |
| `simple_geometry` teacher (step 50) | 32.0% / 76.2% | 99.5% / 100% |
| Slime student (one rollout ahead) | 77.1% / 100% | 99.8% / 100% |
| Miles student (fully async, balanced buffer, staleness 1) | 79.3% / 100% | 99.6% / 100% |
| Prime-RL student (native async, staleness 1) | 55.1% / 93.7% | 97.7% / 100% |

The campaign's README lists further runs.

## Source identity

| Framework | Upstream revision |
|---|---|
| Miles | `8f8e4dff55d80bd9d36c3490c333a119f38480ed` |
| Prime-RL | `550beb6f431b44390afde6b705919765bbe33289` |
| Slime | `4c193f1f37509cca70f0e88807a9305b70f63f4e` |
| verl | `6093e007cc341973c9d9a6fb3867a85976c7c458` (unmodified, no patch; added after the previous experiment) |

Each `frameworks/<framework>/upstream/manifest.json` records the revision, the patch digest, the digest of the original successful-run source archive,
the original run name, and `recorded_campaign_sha256`, the hashes of the campaign files as the previous experiment ran them.
Original archives and raw results were retained separately and are not in this repository.

Miles starts from the revision associated with [PR 3116](https://github.com/radixark/miles/pull/3116).
Its patch requests and retains behavior-policy candidate scores and exposes async launch settings.
The current recipe uses sampled-token OPD, so it does not use the candidate scores. Its other changes are launcher arguments, not patch changes.
Prime-RL's patch selects SDPA for the frozen vision encoder and keeps FA4 for text attention. The aggregate DP3 configuration is only a resolver input.
Slime's patch adds baseline evaluation and timing, handles immutable SGLang arguments, opens and closes required weight-update sessions,
and uses native NCCL groups for the non-offloaded trainer. Its hooks route each task to its teacher.
In every framework, task scores are diagnostic and do not enter the objective.

## Packaging and restructuring

The package replaces original machine paths, node names, and addresses with site values, and takes addresses from the site file.
It replaces the original submission wrappers with a dry-run-first wrapper, and keeps upstream source as pinned revisions plus patches.
After the previous experiment, the code was restructured for readability. Shared code moved to `shared/`, and the preflight tests were removed.
The previous experiment wrote `slime-opd-416`, `mopd-async-401`, and `prime-opd-407`. New runs write `results/<framework>-<job>`.
Most campaign files no longer match `recorded_campaign_sha256`. The pre-restructure package, including the previous data, is at commit `3e27ddb`.
There, files that held site values are templates and do not match, and neither does `teacher-provenance.json`. The other files match.

## Previous experiment: recorded outcomes

The previous experiment used Countdown and Graph Coloring puzzles with thinking off.
It ran 40 updates of 128 samples, learning rate 1e-6, a 256-token response limit, and policy lag 2, on 16 B200 GPUs.
Its teachers were `semianalysisai/Qwen3.6-35B-A3B-countdown-GRPO-20260909` at `237a7f0345e883705026acd7f0745a3637042f73`
and `semianalysisai/Qwen3.6-35B-A3B-graph-color-GRPO-20260909` at `ec87f87177a4ced7256928ce67c438fafa73c28e`.
Scores use 512 development examples per domain from one run per framework.

| Framework | Countdown baseline / final | Graph Coloring baseline / final | Successful allocation |
|---|---|---|---|
| Miles | 10.55% / 45.51% | 26.17% / 88.28% | 21m 34s |
| Prime-RL | 10.94% / 39.65% | 25.78% / 64.65% | 24m 22s |
| Slime | 10.94% / 44.14% | 26.37% / 89.26% | 13m 25s |

Miles then used candidate-based OPD. Allocation time covers startup, training, evaluations, checkpoint work, and cleanup in the successful attempt.
It excludes failed attempts and is not isolated compute or generation throughput.
Prime-RL intermediate evaluations could span changing weights; its baseline and final evaluations used fixed versions.
Slime paused at weight broadcast boundaries, while Miles and Prime-RL scheduled continuously.
The runs differ in objective, scheduler, source mixture, runtime, and checkpoint content. Maximum useful GPU utilization was not demonstrated.

## Data and licenses

`data/` holds reasoning_gym 0.1.25 rows in the default task configuration, with reasoning_gym's default system prompt.
Train rows use seed 20000 and dev rows seed 10000, so the problems are disjoint.
There are 7,788 train rows per task, 126 `caesar_cipher` and 128 `simple_geometry` dev rows, and a 15,576-row mixed training file.
Keep `data/manifest.json` with the data so the inputs can be checked.

Upstream source keeps its original license; a copy accompanies each patch as `upstream/LICENSE`.
Models keep their model licenses, and this package grants no rights to weights.
No license has been assigned to this repository's own code. The owner should select one before public release.
Old experiments and logs remain in prior Git history. This refresh is not a credential audit or history rewrite.
