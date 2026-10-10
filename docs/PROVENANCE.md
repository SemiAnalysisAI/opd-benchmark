# Provenance and interpretation

## Current experiment

The recipe in `shared/recipe.py` runs 20 updates with policy lag 1, thinking on and a 32k context. See the [framework overview](../frameworks/README.md#shared-recipe-sharedrecipepy) for the full settings.

This code has run the single-teacher `caesar_cipher` variant; see the [benchmark report](CAESAR-OPD-2026-09-27.md). It has not run the two-teacher experiment.

## Source identity

| Framework | Upstream revision | Patch |
|---|---|---|
| Miles | `8f8e4dff55d80bd9d36c3490c333a119f38480ed` | Yes |
| NeMo-RL | `19244a0949fb363326f71675223da75e09d14866` (NGC container `nvcr.io/nvidia/nemo-rl:v0.7.0`) | No |
| Prime-RL | `550beb6f431b44390afde6b705919765bbe33289` | Yes |
| Slime | `4c193f1f37509cca70f0e88807a9305b70f63f4e` | Yes |
| verl | `6093e007cc341973c9d9a6fb3867a85976c7c458` | No |

verl and NeMo-RL were added after the previous experiment.

For Miles, Prime-RL and Slime, `frameworks/<framework>/upstream/manifest.json` also records the patch digest, the digest of the original successful-run source archive, the original run name, and `recorded_campaign_sha256` (hashes of the campaign files as the previous experiment ran them). The original archives and raw results were kept separately and are not in this repository.

What the patches do:

- **Miles** starts from the revision associated with [PR 3116](https://github.com/radixark/miles/pull/3116). Its patch requests and keeps behavior-policy candidate scores and exposes async launch settings. The current recipe uses sampled-token OPD, so it does not use the candidate scores. Its other changes are launcher arguments, not patch changes.
- **Prime-RL**'s patch selects SDPA for the frozen vision encoder and keeps FA4 for text attention. The aggregate DP3 configuration is only a resolver input.
- **Slime**'s patch adds baseline evaluation and timing, handles immutable SGLang arguments, opens and closes the required weight-update sessions, and uses native NCCL groups for the non-offloaded trainer. Its hooks route each task to its teacher.

In every framework, task scores are diagnostic and do not enter the objective.

## Packaging and restructuring

The package replaces the original machine paths, node names and addresses with values from the site file. It replaces the original submission wrappers with a dry-run-first wrapper, and keeps upstream source as pinned revisions plus patches.

After the previous experiment, the code was restructured for readability: shared code moved to `shared/` and the preflight tests were removed. The previous experiment wrote `slime-opd-416`, `mopd-async-401` and `prime-opd-407`; new runs write `results/<framework>-<job>`.

Most campaign files no longer match `recorded_campaign_sha256`. The pre-restructure package, including the previous data, is at commit `3e27ddb`. There, the files that held site values are templates and do not match, and neither does `teacher-provenance.json`. The other files match.

## Previous experiment

The previous experiment used Countdown and Graph Coloring puzzles with thinking off: 40 updates of 128 samples, learning rate 1e-6, a 256-token response limit and policy lag 2, on 16 B200 GPUs. Its teachers were:

- `semianalysisai/Qwen3.6-35B-A3B-countdown-GRPO-20260909` at `237a7f0345e883705026acd7f0745a3637042f73`
- `semianalysisai/Qwen3.6-35B-A3B-graph-color-GRPO-20260909` at `ec87f87177a4ced7256928ce67c438fafa73c28e`

Scores use 512 development examples per domain, from one run per framework.

| Framework | Countdown baseline / final | Graph Coloring baseline / final | Successful allocation |
|---|---|---|---|
| Miles | 10.55% / 45.51% | 26.17% / 88.28% | 21m 34s |
| Prime-RL | 10.94% / 39.65% | 25.78% / 64.65% | 24m 22s |
| Slime | 10.94% / 44.14% | 26.37% / 89.26% | 13m 25s |

How to read these numbers:

- Miles used candidate-based OPD in this experiment.
- Allocation time covers startup, training, evaluations, checkpoint work and cleanup in the successful attempt. It excludes failed attempts and does not measure compute or generation throughput.
- Prime-RL's intermediate evaluations could span changing weights; its baseline and final evaluations used fixed versions.
- Slime paused at weight-broadcast boundaries; Miles and Prime-RL scheduled continuously.
- The runs differ in objective, scheduler, source mixture, runtime and checkpoint content. Maximum useful GPU utilization was not demonstrated.

## Data and licenses

`data/` holds reasoning_gym 0.1.25 rows in the default task configuration, with reasoning_gym's default system prompt. Train rows use seed 20000 and dev rows seed 10000, so the problems are disjoint. There are 7,788 train rows per task, 126 `caesar_cipher` and 128 `simple_geometry` dev rows, and a 15,576-row mixed training file. Keep `data/manifest.json` with the data so the inputs can be checked.

Upstream source keeps its original license; a copy sits next to each patch as `upstream/LICENSE`. Models keep their own licenses, and this package grants no rights to weights.

No license has been assigned to this repository's own code yet. The owner should choose one before public release.

Old experiments and logs remain in Git history. This refresh was not a credential audit or a history rewrite.
