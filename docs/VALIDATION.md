# Validation

## Current code

The package ran the single-teacher `caesar_cipher` variant in Miles, Prime-RL, Slime and verl on 16 B200 GPUs on September 27, 2026; every run completed 20 updates and was benchmarked ([report](CAESAR-OPD-2026-09-27.md)). The two-teacher MOPD recipe has not run. The checks below need no GPU.

```bash
python3 -m unittest discover -s tests -v
```

The package tests check the following:
- Campaign files compile and contain no machine paths.
- Dataset digests and row counts match.
- Patch digests match.
- Invalid site files are rejected.
- Existing output directories are preserved.
- Submission defaults to a dry run.
- Telemetry discovers the announced engine ports.
- Hosted credentials are read as configured.
- The scorer requires the answer after `</think>`. This test is skipped unless `reasoning-gym==0.1.25` is installed.

The hosted campaigns have their own offline tests; see their READMEs.
There is no automated preflight. Distributed training, checkpoint conversion, teacher fusing on real weights,
image reconstruction, and other GPU types remain untested.

## Historical record: September 15, 2026 packaging checks

These checks covered the previous experiment's package, before the restructure. No cluster job was submitted.

Six package tests passed under Python 3.12.
Every patch applied to a clean archive of its pinned revision, and each changed file matched the original successful-run archive by SHA-256.
Miles had four changed files, Prime-RL one, and Slime five. Repeat this check with local upstream clones:

```bash
python3 tools/verify_sources.py --miles /path/to/miles --prime-rl /path/to/prime-rl --slime /path/to/slime
```

A Slime campaign was prepared from a local clone, and its submission command printed the expected two-node, 16-GPU request without invoking Slurm.
A scan for token patterns, private keys, and original private addresses and paths found none.
That scan was not an audit of historical commits.
