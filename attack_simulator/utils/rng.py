import inspect
import logging
import random
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("trainer")


def get_tweaked_rng(seed: Optional[int] = None) -> Tuple[np.random.Generator, int]:
    new_seed = np.random.SeedSequence(None).entropy if seed is None else seed

    caller = "UNKNOWN"
    frame = inspect.currentframe()
    while frame:
        if "self" in frame.f_locals:
            caller = frame.f_locals["self"].__class__.__name__
            break
        frame = frame.f_back

    assert isinstance(new_seed, int)

    # tweak passed seeds for different callers
    tweaked = new_seed + int.from_bytes(bytes(caller.encode()), byteorder="big")

    logger.info(
        "%s uses RNG primed with %d\n"
        "derived from reproducible seed %d\n"
        'combined with the string "%s"',
        caller,
        tweaked,
        seed,
        caller,
    )

    return np.random.default_rng(tweaked), new_seed


def get_rng(seed: Optional[int] = None) -> Tuple[np.random.Generator, int]:
    if seed is None:
        new_seed = int(np.random.SeedSequence(None).entropy)
        long_seed = np.uint64(0)
        long_seed = np.bitwise_or(long_seed, new_seed)
    else:
        long_seed = seed

    return np.random.default_rng(long_seed), long_seed


def set_seeds_from_bytes(seed: Optional[int] = None) -> Tuple[int, int]:
    rng, new_seed = get_rng(seed)
    random_bytes = rng.bytes(8)

    random.seed(random_bytes)
    np.random.seed(int.from_bytes(random_bytes[:4], "big"))

    entropy = np.random.SeedSequence(None).entropy
    assert isinstance(entropy, int)

    return new_seed, entropy


def set_seeds(seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)
    return seed
