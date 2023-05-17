import pytest
from attack_simulator import AGENT_DEFENDER
from attack_simulator.env.env import AttackSimulationEnv

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
def test_sim_defender_actions(env: AttackSimulationEnv, defender_class) -> None:
    done = False
    obs, info = env.reset()

    defender = defender_class(env.observation_space[AGENT_DEFENDER], env.action_space[AGENT_DEFENDER], {})
