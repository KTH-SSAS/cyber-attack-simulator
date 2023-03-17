import pytest

from attack_simulator.optimal_defender import TripwirePolicy
from attack_simulator.random_defender import RandomPolicy
from attack_simulator.sim import AttackSimulator

NUM_TRIALS = 1000


@pytest.mark.parametrize(
    "defender_class",
    [
        RandomPolicy,
        TripwirePolicy,
    ],
)
def test_sim_defender_actions(simulator: AttackSimulator, defender_class) -> None:
    done = False
    obs, _ = simulator.reset()

    total_ttc = simulator.ttc_total
