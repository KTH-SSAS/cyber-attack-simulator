import inspect
import logging
import random

import numpy as np
import torch

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


def set_seeds(seed=None):
    same = seed is not None and seed < 0

    if same:
        seed = abs(seed)

    rng, seed = get_rng(seed)
    random_bytes = rng.bytes(8)
    random.seed(random_bytes)
    np.random.seed(int.from_bytes(random_bytes[:4], "big"))
    torch.manual_seed(int.from_bytes(random_bytes, "big"))

    return seed, seed if same else np.random.SeedSequence(None).entropy
