import inspect
import logging
import random

import numpy as np
import torch

logger = logging.getLogger("trainer")


def get_rng(seed=None):
    if seed is None:  # fall back to real entropy
        seed = np.random.SeedSequence(None).entropy

    caller = "UNKNOWN"
    frame = inspect.currentframe()
    while frame:
        if "self" in frame.f_locals:
            caller = frame.f_locals["self"].__class__.__name__
            break
        frame = frame.f_back

    # tweak passed seeds for different callers
    tweaked = seed + int.from_bytes(bytes(caller.encode()), byteorder="big")

    logger.info(
        f"{caller} uses RNG primed with {tweaked}\n"
        f"    derived from reproducible seed {seed}\n"
        f'    combined with the string "{caller}"'
    )

    return np.random.default_rng(tweaked), seed


def set_seeds(seed=None):
    rng, seed = get_rng(seed)
    random_bytes = rng.bytes(8)
    random.seed(random_bytes)
    np.random.seed(int.from_bytes(random_bytes[:4], "big"))
    torch.manual_seed(int.from_bytes(random_bytes, "big"))

    return seed, np.random.SeedSequence(None).entropy
