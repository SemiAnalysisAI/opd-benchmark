"""Zero task reward: MOPD's teacher KL is the only optimization signal."""


def compute_score(*args, **kwargs) -> float:
    return 0.0

