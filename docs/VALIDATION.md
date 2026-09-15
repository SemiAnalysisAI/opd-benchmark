# Package validation

The package was checked locally on September 15, 2026. No cluster job was submitted during packaging.

The following six tests passed under Python 3.12.

```bash
python3 -m unittest discover -s tests -v
```

The tests render and parse every campaign template, check every dataset digest and row count,
verify patch digests, reject invalid site values, preserve existing output directories, and confirm that submission defaults to a dry run.

All framework patches were applied to clean archives of their pinned upstream revisions.
Every changed source file matched its counterpart from the original successful-run source archive by SHA-256.
Miles had four changed files, Prime-RL had one, and Slime had five.
The check can be repeated with local upstream Git clones as follows.

```bash
python3 tools/verify_sources.py --miles /path/to/miles --prime-rl /path/to/prime-rl --slime /path/to/slime
```

A complete Slime campaign was prepared in a temporary directory from a local clone.
It checked out the correct revision, applied the patch, expanded site values, and materialized the datasets.
Its submission command printed the expected two-node, 16-GPU allocation request and did not invoke Slurm.

The new package was scanned for common Hugging Face and GitHub token patterns, private-key headers,
original private network addresses, and original personal or cluster paths. No such values were found in the new package.
This targeted scan is not a security audit of the repository's historical commits.

Distributed training, new checkpoint conversion, runtime image reconstruction, and execution on a different GPU remain untested for this package.
The original experiment outcomes do not remove these validation gaps.
