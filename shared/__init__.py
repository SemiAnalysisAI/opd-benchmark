"""Code shared by every framework and hosted provider.

recipe       the experiment settings and pinned models
puzzles      packaged datasets and their checksums
scoring      the puzzle verifier
slurm        campaign paths, process supervision and the Slurm allocation
container    the two-node Pyxis runtime used by Miles and Slime
capture      telemetry sampler (GPU, host and Prometheus metrics)
credentials  API keys for the hosted providers
"""
