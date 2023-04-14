import random

import numpy as np
import pytest
import torch

from attack_simulator.rng import get_rng, set_seeds


@pytest.mark.parametrize("seed", [42, None])
def test_rng_get_rng(seed):
    rng0, seed0 = get_rng(seed)
    assert seed is None or seed0 == seed

    samples0 = [np.array((rng.uniform(), rng.bytes(8), rng.choice(range(11)))) for rng in [rng0]][0]

    rng1, seed1 = get_rng(seed0)
    assert seed0 == seed1

    samples1 = [np.array((rng.uniform(), rng.bytes(8), rng.choice(range(11)))) for rng in [rng1]][0]

    assert np.all(samples0 == samples1)


@pytest.mark.parametrize("seed", [42, 43])
def test_rng_set_seeds(seed):
    set_seeds(seed)

    u = torch.tensor([1 / 11] * 11)
    samples0 = np.array(
        (
            random.uniform(0, 1),
            random.getrandbits(64),
            random.choice(range(11)),
            np.random.uniform(),
            np.random.bytes(8),
            np.random.choice(range(11)),
            torch.distributions.Uniform(0, 1).sample().item(),
            bytes(torch.randint(256, (8,))),
            u.multinomial(1).item(),
        )
    )

    set_seeds(seed)

    samples1 = np.array(
        (
            random.uniform(0, 1),
            random.getrandbits(64),
            random.choice(range(11)),
            np.random.uniform(),
            np.random.bytes(8),
            np.random.choice(range(11)),
            torch.distributions.Uniform(0, 1).sample().item(),
            bytes(torch.randint(256, (8,))),
            u.multinomial(1).item(),
        )
    )

    assert np.all(samples0 == samples1)
