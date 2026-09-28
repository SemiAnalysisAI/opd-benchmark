# Validation

## What has run on GPUs

On September 27, 2026 the package ran the single-teacher `caesar_cipher` variant in Miles, Prime-RL, Slime and verl on 16 B200 GPUs. Every run completed 20 updates and was benchmarked; see the [report](CAESAR-OPD-2026-09-27.md). NeMo-RL deadlocked before its first update. The two-teacher MOPD recipe has not run.

## Package tests

These need no GPU:

```bash
python3 -m unittest discover -s tests -v
```

They check that:

- campaign files compile and contain no machine paths;
- dataset digests and row counts match;
- patch digests match;
- invalid site files are rejected;
- existing output directories are preserved;
- submission defaults to a dry run;
- telemetry finds the announced engine ports;
- hosted credentials are read as configured;
- Miles, Prime-RL, Slime and verl launch configurations enable thinking;
- the scorer requires the answer after `</think>` (skipped unless `reasoning-gym==0.1.25` is installed).

The hosted campaigns have their own offline tests; see their READMEs.

## Source patches

`tools/verify_sources.py` applies each patch to a clean archive of its pinned revision and checks every changed file against the SHA-256 recorded in the manifest. verl and NeMo-RL run unmodified and have nothing to check.

```bash
python3 tools/verify_sources.py --miles /path/to/miles --prime-rl /path/to/prime-rl --slime /path/to/slime
```

## Not covered

There is no automated preflight. Distributed training, checkpoint conversion, teacher fusing on real weights, image reconstruction and other GPU types are not covered by these tests.

## September 15, 2026 packaging checks

These checks covered the previous experiment's package, before the restructure. No cluster job was submitted.

- Six package tests passed under Python 3.12.
- Every patch applied to a clean archive of its pinned revision, and each changed file matched the original successful-run archive by SHA-256. Miles had four changed files, Prime-RL one and Slime five.
- A Slime campaign was prepared from a local clone. Its submission command printed the expected two-node, 16-GPU request without invoking Slurm.
- A scan for token patterns, private keys, and the original private addresses and paths found none. It did not cover historical commits.
