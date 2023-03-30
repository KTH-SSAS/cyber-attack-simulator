from collections import OrderedDict

import numpy as np
import pytest

from attack_simulator.agents import (  # InformedAttacker,; RandomNoActionAttacker,; RoundRobinNoActionAttacker,
    Agent,
    PathFinderAttacker,
    RandomAttacker,
    RoundRobinAttacker,
)
from attack_simulator.agents.attackers.searchers import BreadthFirstAttacker, DepthFirstAttacker
from attack_simulator import ACTION_TERMINATE, ACTION_WAIT, AGENT_ATTACKER
from attack_simulator.env.env import get_agent_obs
from attack_simulator.sim import Simulator


@pytest.mark.parametrize(
    "attacker_class",
    [
        PathFinderAttacker,
        # InformedAttacker,
        # RandomNoActionAttacker,
        RandomAttacker,
        RoundRobinAttacker,
        # RoundRobinNoActionAttacker,
        BreadthFirstAttacker,
        DepthFirstAttacker,
    ],
)
def test_sim_attacker_actions(simulator: Simulator, attack_graph, attacker_class) -> None:
    done = False
    obs, info = simulator.reset()

    total_ttc = simulator.ttc_total

    attacker: Agent = attacker_class(
        dict(
            attack_graph=attack_graph,
            num_special_actions=2,
            terminate_action=ACTION_TERMINATE,
            wait_action=ACTION_WAIT,
            seed=42,
        )
    )

    last_ttc_sum = simulator.ttc_total
    while info.time <= total_ttc and not done:

        obs_dict = get_agent_obs(obs)[AGENT_ATTACKER]
        action = attacker.compute_action_from_dict(obs_dict)
        assert action != ACTION_TERMINATE
        assert action != ACTION_WAIT

        attack_surface = obs_dict["action_mask"].reshape(-1)[2:]
        assert all(attack_surface == obs.attack_surface)
        valid_actions = np.flatnonzero(attack_surface)
        assert action - 2 in valid_actions

        obs, info = simulator.step(OrderedDict([(AGENT_ATTACKER, action)]))

        done = not any(obs.attack_surface)
        assert simulator.ttc_total < last_ttc_sum
        last_ttc_sum = simulator.ttc_total

    assert done, "Attacker failed to explore all attack steps"


# def test_agents_attackers_informed(simulator: AttackSimulator) -> None:
#     a = InformedAttacker(dict(attack_graph=simulator.g))

#     while not a.done:
#         obs = simulator.attacker_observation()
#         a = a.act(obs)
#         simulator.attacker_action(a)


# def test_agents_attackers_random_no_action():
#     a = RandomNoActionAttacker(dict(random_seed=42))
#     for size in np.random.randint(1, 99, size=4):
#         obs = np.zeros(size)
#         ones = np.random.randint(size, size=2)
#         obs[ones] = 1
#         for _ in range(5):
#             action = a.act(obs) - 1
#             assert action == -1 or action in ones


# def test_agents_attackers_random():
#     a = RandomAttacker(dict(random_seed=42))
#     for size in np.random.randint(1, 99, size=4):
#         obs = np.zeros(size)
#         ones = np.random.randint(size, size=2)
#         obs[ones] = 1
#         for _ in range(5):
#             assert (a.act(obs) - 1) in ones

#     assert a.update(obs, 0, True) is None


# def test_agents_attackers_round_robin_no_action():
#     a = RoundRobinNoActionAttacker({})
#     for size in np.random.randint(1, 99, size=4):
#         obs = np.full(size, 1)
#         zeros = np.random.randint(size, size=size // 3)
#         obs[zeros] = 0
#         last_action = a.act(obs)
#         max_action = max(np.flatnonzero(obs)) + 1
#         for _ in range(size + size):
#             action = a.act(obs)
#             assert (last_action < action or last_action == max_action) and (action - 1) not in zeros
#             last_action = action

#     assert a.update(obs, 0, True) is None


# def test_agents_attackers_round_robin():
#     a = RoundRobinAttacker({})
#     for size in np.random.randint(1, 99, size=4):
#         obs = np.full(size, 1)
#         zeros = np.random.randint(size, size=size // 3)
#         obs[zeros] = 0
#         last_action = a.act(obs) - 1
#         max_action = max(np.flatnonzero(obs))
#         for _ in range(size + size):
#             action = a.act(obs) - 1
#             assert (last_action < action or last_action == max_action) and action not in zeros
#             last_action = action

#     assert a.update(obs, 0, True) is None


def test_attackers_breadth_first() -> None:
    pass


def test_attackers_depth_first() -> None:
    pass
