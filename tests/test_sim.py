from collections import OrderedDict
from pathlib import Path

import pytest

from attack_simulator.agents.agent import Agent
from attack_simulator.agents.searchers import BreadthFirstAttacker
from attack_simulator.config import EnvConfig
from attack_simulator.constants import ACTION_TERMINATE, ACTION_WAIT, AGENT_ATTACKER, AGENT_DEFENDER
from attack_simulator.graph import AttackGraph
from attack_simulator.observation import Info, Observation
from attack_simulator.rust_wrapper import rust_sim_init
from attack_simulator.sim import AttackSimulator


def test_sim_reset(simulator: AttackSimulator) -> None:
    obs, info = simulator.reset()
    assert info.time == 0


@pytest.mark.skip(reason="Not implemented")
def test_sim_defender_action(simulator: AttackSimulator) -> None:
    pass


@pytest.mark.skip(reason="Not implemented")
def test_sim_attacker_action(simulator: AttackSimulator) -> None:
    pass


def test_sim_step(simulator) -> None:
    # Take a step
    actions = {AGENT_ATTACKER: ACTION_WAIT, AGENT_DEFENDER: ACTION_WAIT}
    obs: Observation
    info: Info
    obs, info = simulator.step(actions)

    # Check that the time has advanced
    assert info.time == 1


def test_rust_and_python():
    pass

    config_yaml = Path("config/test_config.yaml")
    config: EnvConfig = EnvConfig.from_yaml(config_yaml)

    graph = AttackGraph(config.graph_config)

    sim_config = config.sim_config.replace(randomize_ttc=False)

    rust_sim = rust_sim_init(sim_config, graph)
    python_sim = AttackSimulator(sim_config, graph)

    rust_obs: Observation
    rust_info: Info
    python_obs: Observation
    python_info: Info

    # Reset both sims
    rust_obs, rust_info = rust_sim.reset()
    python_obs, python_info = python_sim.reset()

    def compare_obs(rust_obs, python_obs):
        # Check that the observations are the same
        compare_fields(rust_obs.attack_state, python_obs.attack_state)
        compare_fields(rust_obs.defense_state, python_obs.defense_state)
        compare_fields(rust_obs.attack_surface, python_obs.attack_surface)
        # compare_fields(rust_obs.ids_observation, python_obs.ids_observation)
        compare_fields(rust_obs.ttc_remaining, python_obs.ttc_remaining)

    def compare_fields(rust, python):
        for i, (x1, x2) in enumerate(zip(rust, python)):
            assert x1 == x2, f"Failed at index {i}: rust: {x1} != python: {x2}"

    config_dict = dict(
        num_special_actions=2,
        terminate_action=ACTION_TERMINATE,
        wait_action=ACTION_WAIT,
        seed=0,
    )

    rust_attacker: Agent = BreadthFirstAttacker(config_dict)
    python_attacker: Agent = BreadthFirstAttacker(config_dict)

    while rust_info.time == python_info.time and rust_info.time < graph.ttc_params.sum() - 1:

        compare_obs(rust_obs, python_obs)

        assert rust_info.num_compromised_steps == python_info.num_compromised_steps
        assert rust_info.num_observed_alerts == python_info.num_observed_alerts
        assert rust_info.num_compromised_flags == python_info.num_compromised_flags
        assert rust_info.perc_compromised_flags == python_info.perc_compromised_flags
        assert rust_info.perc_compromised_steps == python_info.perc_compromised_steps
        assert rust_info.perc_defenses_activated == python_info.perc_defenses_activated
        assert rust_info.time == python_info.time

        # Take a step

        # Get the actions
        rust_action = rust_attacker.compute_action_from_dict(
            {"attack_surface": rust_obs.attack_surface}
        )
        python_action = python_attacker.compute_action_from_dict(
            {"attack_surface": python_obs.attack_surface}
        )
        rust_obs, rust_info = rust_sim.step(OrderedDict([(AGENT_ATTACKER, rust_action)]))
        python_obs, python_info = python_sim.step(OrderedDict([(AGENT_ATTACKER, python_action)]))

    assert sum(rust_obs.ttc_remaining) == 0
    assert sum(python_obs.ttc_remaining) == 0


def test_determinism(simulator: AttackSimulator):
    def compare_fields(v1, v2):
        for i, (x1, x2) in enumerate(zip(v1, v2)):
            assert x1 == x2, f"Failed at index {i}: initial: {x1} != current: {x2}"

    first_obs, first_info = simulator.reset()

    for _ in range(100):
        obs, info = simulator.reset()
        compare_fields(first_obs.attack_state, obs.attack_state)
        compare_fields(first_obs.ttc_remaining, obs.ttc_remaining)
