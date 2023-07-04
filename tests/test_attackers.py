from collections import OrderedDict

import numpy as np
import pytest

from attack_simulator import ACTION_TERMINATE, ACTION_WAIT, AGENT_ATTACKER
from attack_simulator.agents import (  # InformedAttacker,; RandomNoActionAttacker,; RoundRobinNoActionAttacker,
    Agent,
    PathFinderAttacker,
    RandomAttacker,
    RoundRobinAttacker,
)
from attack_simulator.agents.attackers.searchers import BreadthFirstAttacker, DepthFirstAttacker
from attack_simulator.env.env import AttackSimulationEnv

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
def test_attacker_actions(env: AttackSimulationEnv, attack_graph, attacker_class) -> None:
    done = False

    obs, info = env.reset()
    
    total_ttc = info[AGENT_ATTACKER]["sum_ttc_remaining"]

    num_special_actions = env.num_special_actions

    attacker: Agent = attacker_class(
        dict(
            attack_graph=attack_graph,
            seed=42,
        )
    )

    last_ttc_remaining = total_ttc
    steps = 0
    sum_rewards = 0
    while steps <= total_ttc and not done:

        action = attacker.compute_action_from_dict(obs[AGENT_ATTACKER])
        assert action != ACTION_TERMINATE
        assert action != ACTION_WAIT

        attack_surface = obs[AGENT_ATTACKER]["action_mask"].reshape(-1)[num_special_actions:]
        #assert all(attack_surface == obs.attack_surface)
        valid_actions = np.flatnonzero(attack_surface)
        assert action - num_special_actions in valid_actions

        obs, rewards, terminated, truncated, info  = env.step({AGENT_ATTACKER: action})

        sum_rewards += rewards[AGENT_ATTACKER]

        done = terminated[AGENT_ATTACKER] or truncated[AGENT_ATTACKER]
        # Check that the attacker is reducing overall time to compromise
        assert info[AGENT_ATTACKER]["sum_ttc_remaining"] < last_ttc_remaining
        last_ttc_remaining = info[AGENT_ATTACKER]["sum_ttc_remaining"]
        steps += 1
    assert sum_rewards == sum(env.state.attack_step_rewards)
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
