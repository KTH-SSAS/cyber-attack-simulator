import random

import numpy as np
import pytest


from attack_simulator.utils.rng import get_rng, set_seeds


@pytest.mark.parametrize("seed", [42, None])
def test_rng_get_rng(seed):
    rng0, seed0 = get_rng(seed)
    assert seed is None or seed0 == seed

    samples0 = [np.array((rng.uniform(), rng.bytes(8), rng.choice(range(11)))) for rng in [rng0]][0]

    rng1, seed1 = get_rng(seed0)
    assert seed0 == seed1

    samples1 = [np.array((rng.uniform(), rng.bytes(8), rng.choice(range(11)))) for rng in [rng1]][0]

    assert np.all(samples0 == samples1)

