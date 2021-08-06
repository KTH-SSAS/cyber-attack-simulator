import inspect
import logging

import numpy as np

logger = logging.getLogger("trainer")


def get_rng(seed=None):
    if seed is None:  # fall back to real entropy
        seed = np.random.SeedSequence(None).entropy

    if seed < 0:
        seed = -seed

    try:  # whose calling?
        caller = inspect.currentframe().f_back.f_locals["self"].__class__.__name__
    except Exception:
        caller = "UNKNOWN"

    # tweak passed seeds for different callers
    tweaked = seed + int.from_bytes(bytes(caller.encode()), byteorder="big")

    logger.info(
        f"{caller} uses RNG primed with {tweaked}\n"
        f"    derived from reproducible seed {seed}\n"
        f'    combined with the string "{caller}"'
    )

    return np.random.default_rng(tweaked), seed
