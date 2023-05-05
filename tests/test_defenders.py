import pytest

from attack_simulator.mal.sim import Simulator
from attack_simulator.rllib.random_defender import RandomPolicy
from attack_simulator.rllib.tripwire_defender import TripwirePolicy

NUM_TRIALS = 1000


@pytest.mark.parametrize(
    "defender_class",
    [
        RandomPolicy,
        TripwirePolicy,
    ],
)
def test_sim_defender_actions(simulator: Simulator, defender_class) -> None:
    done = False
    obs, _ = simulator.reset()

    total_ttc = simulator.ttc_total
