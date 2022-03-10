import inspect
import logging
import random

import numpy as np
import torch

logger = logging.getLogger("trainer")


def get_tweaked_rng(seed=None):
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
        "%s uses RNG primed with %d\n"
        "derived from reproducible seed %d\n"
        'combined with the string "%s"',
        caller,
        tweaked,
        seed,
        caller,
    )

    return np.random.default_rng(tweaked), seed


def get_rng(seed):
    if seed is None:  # fall back to real entropy
        seed = np.random.SeedSequence(None).entropy
    return np.random.default_rng(seed), seed


def set_seeds_from_bytes(seed=None):

    rng, seed = get_rng(seed)
    random_bytes = rng.bytes(8)

    random.seed(random_bytes)
    np.random.seed(int.from_bytes(random_bytes[:4], "big"))
    torch.manual_seed(int.from_bytes(random_bytes, "big"))

    return seed, np.random.SeedSequence(None).entropy


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed
