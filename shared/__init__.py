"""Shared on-policy distillation (OPD) code, used by every framework and hosted provider.

recipe       the fixed experiment and pinned models
puzzles      packaged datasets and their checksums
scoring      the puzzle verifier
slurm        prepared-campaign paths, process supervision and the Slurm allocation
container    the two-node container runtime shared by Miles and Slime
capture      the one telemetry process (GPU, host and Prometheus samples)
credentials  API keys for the hosted providers
"""
