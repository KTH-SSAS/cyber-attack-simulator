import dataclasses

import numpy as np
import pytest

from attack_simulator.config import EnvConfig
from attack_simulator.env import AttackSimulationEnv
from attack_simulator.graph import AttackGraph
from attack_simulator.renderer import AttackSimulationRenderer
from attack_simulator.sim import AttackSimulator


def test_sim_noisy_observation(simulator: AttackSimulator) -> None:

    # Set observations to be very noisy
    simulator.false_negative = 0.5
    simulator.false_positive = 0.5

    obs1 = simulator.observe()
    obs2 = simulator.observe()

    # If not time passes, the observations do not change
    assert all(obs1 == obs2)

    # When time changes, so will the observation
    simulator.step()

    obs2 = simulator.observe()

    assert any(obs1 != obs2)


def test_sim_interpretations(simulator: AttackSimulator) -> None:
    obs = simulator.observe()
    i_obs = simulator.interpret_observation(obs)
    i_def = simulator.interpret_defenses(simulator.defense_state)
    i_att = simulator.interpret_attacks(simulator.attack_state)
    assert i_obs == (i_def, i_att)


def test_sim_defender_action(simulator: AttackSimulator) -> None:
    pass


def test_sim_attacker_action(simulator: AttackSimulator) -> None:
    pass


def test_sim_initial_state(simulator: AttackSimulator) -> None:
    obs = np.array(simulator.observe())
    assert all(obs == [1] * simulator.g.num_defenses + [0] * simulator.g.num_attacks)


def test_sim_observe(simulator: AttackSimulator) -> None:
    obs = np.array(simulator.observe())
    assert all(obs == [1] * simulator.g.num_defenses + [0] * simulator.g.num_attacks)
