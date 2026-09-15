# Provenance and interpretation

## Source identity

| Framework | Upstream revision | Execution |
|---|---|---|
| Miles | `8f8e4dff55d80bd9d36c3490c333a119f38480ed` | Fully async candidate-based MOPD with a balanced consumer buffer. |
| Prime-RL | `550beb6f431b44390afde6b705919765bbe33289` | Native asynchronous per-source OPD through reference KL. |
| Slime | `4c193f1f37509cca70f0e88807a9305b70f63f4e` | Native batched asynchronous OPD through an advantage correction. |

Each `manifest.json` records the SHA-256 digest of the original successful-run source archive.
It also records hashes of the original campaign files before template substitutions and the exact framework patch digest.
Original archives and raw results were retained separately. They are not copied into this repository.

Miles starts from the preserved revision associated with [PR 3116](https://github.com/radixark/miles/pull/3116).
Its patch requests and retains behavior-policy candidate scores in the class-based generator and exposes async launch settings.
The candidate objective and teacher routing remain those of the pinned revision.
The campaign supplies a bounded buffer that consumes 64 active samples per domain per update and rejects excessive staleness.
It also retries transient frozen-teacher transport errors with a bounded attempt count.

Prime-RL retains its native OPD objective. Its patch selects SDPA for the frozen vision encoder while retaining FA4 for text attention.
The campaign launches three independent TP2 policy engines and a native router.
The aggregate DP3 configuration is a resolver input; it is not the executable engine configuration.

Slime retains its native OPD objective. Its patch adds baseline evaluation and timing, handles immutable SGLang arguments,
opens and closes required weight-update sessions, and uses native NCCL groups for the non-offloaded trainer.
Its reward hook routes each puzzle domain to the corresponding frozen teacher.
Task rewards remain diagnostic and do not enter the OPD objective.

## Packaging changes

This package replaces original machine paths, node names, and addresses with explicit site values.
It selects network addresses from that site file rather than inferring them from a public-route socket.
It passes the configured Megatron source path to the Miles launcher.
It replaces the original submission wrappers with a dry-run-first wrapper that requires `--submit`.
It compresses the unchanged datasets and checks their original byte hashes after decompression.
It keeps upstream source as pinned Git revisions plus patches rather than vendoring whole repositories.
Generated source archives exclude Git metadata and virtual environments. Dataset and training source contents remain included.

These changes have passed local package checks. They have not received a new GPU run.
Original runtime failures and compatibility repairs are documented in the framework patches and setup requirements.
The operator must not treat local syntax checks as a substitute for distributed runtime validation.

## Recorded outcomes

All experiments completed 40 updates with 128 consumed samples per update on 16 B200 GPUs.
The following scores use 512 development examples per domain and a 256-token response limit.
They are results from one run per framework and are not new held-out test results.

| Framework | Countdown baseline / final | Graph Coloring baseline / final | Successful allocation |
|---|---|---|---|
| Miles | 10.55% / 45.51% | 26.17% / 88.28% | 21m 34s |
| Prime-RL | 10.94% / 39.65% | 25.78% / 64.65% | 24m 22s |
| Slime | 10.94% / 44.14% | 26.37% / 89.26% | 13m 25s |

Allocation time includes startup, training, evaluations, checkpoint work, and cleanup in the successful attempt.
It excludes earlier failed attempts. It is not isolated trainer compute time or pure generation throughput.
The runs differ in objective, scheduler, source mixture, runtime, and checkpoint content.
Prime-RL intermediate evaluations can span changing weights; its baseline and final evaluations used fixed versions.
Slime pauses at weight broadcast boundaries, while Miles and Prime-RL use continuous async scheduling.
All runs bounded accepted policy lag at two versions, using each campaign's recorded version convention.
Maximum useful GPU utilization was not demonstrated.

## Data and licenses

The files under `data/` are the exact synthetic Countdown and Graph Coloring inputs used in the recorded experiments.
They contain 10,000 training examples per domain, 512 development examples per domain, and separate test files.
The mixed training file contains 20,000 rows. The development split was used for the reported scores.
The pinned Miles source contains the puzzle preparation and scoring implementation under `examples/mopd_puzzles/`.
Preserve `data/manifest.json` when copying the data so the exact inputs can be checked.

Upstream source remains subject to its original license. Copies of the top-level upstream license files accompany each patch.
Hugging Face models remain subject to their model licenses. This package does not grant additional rights to model weights.
No new blanket license has been assigned to this repository's campaign code.
The repository owner should select its distribution license before public release.

The old experiments and logs were removed from the current layout, but remain recoverable in prior Git history.
This refresh does not constitute a credential audit of historical commits or a history rewrite.
