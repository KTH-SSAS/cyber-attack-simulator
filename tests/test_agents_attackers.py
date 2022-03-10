import numpy as np

from attack_simulator.agents import (
    InformedAttacker,
    PathFinderAttacker,
    RandomAttacker,
    RandomNoActionAttacker,
    RoundRobinAttacker,
    RoundRobinNoActionAttacker,
)
from attack_simulator.sim import AttackSimulator

from .test_tests_utils import np_bits


def test_agents_attackers_planning(simulator: AttackSimulator):

    # n = len(simulator.g.attack_steps)
    a = PathFinderAttacker(dict(simulator=simulator))

    done = False

    while not done:
        attack_index = a.act(simulator.attack_surface) - 1
        done = a.done

        if done:
            break

        assert -1 <= attack_index < simulator.num_attack_steps

        if attack_index != -1:
            done = simulator.attack_action(attack_index)

    assert True


def test_agents_attackers_informed(attack_graph):
    a = InformedAttacker(dict(attack_graph=attack_graph))
    n = len(attack_graph.attack_steps)

    for i in range(1, 1 << n):
        obs = np_bits(i, size=n)
        assert (a.act(obs) - 1) in np.flatnonzero(obs)


def test_agents_attackers_random_no_action():
    a = RandomNoActionAttacker(dict(random_seed=42))
    for size in np.random.randint(1, 99, size=4):
        obs = np.zeros(size)
        ones = np.random.randint(size, size=2)
        obs[ones] = 1
        for _ in range(5):
            action = a.act(obs) - 1
            assert action == -1 or action in ones


def test_agents_attackers_random():
    a = RandomAttacker(dict(random_seed=42))
    for size in np.random.randint(1, 99, size=4):
        obs = np.zeros(size)
        ones = np.random.randint(size, size=2)
        obs[ones] = 1
        for _ in range(5):
            assert (a.act(obs) - 1) in ones

    assert a.update(obs, 0, True) is None


def test_agents_attackers_round_robin_no_action():
    a = RoundRobinNoActionAttacker()
    for size in np.random.randint(1, 99, size=4):
        obs = np.full(size, 1)
        zeros = np.random.randint(size, size=size // 3)
        obs[zeros] = 0
        last_action = a.act(obs)
        max_action = max(np.flatnonzero(obs)) + 1
        for _ in range(size + size):
            action = a.act(obs)
            assert (last_action < action or last_action == max_action) and (action - 1) not in zeros
            last_action = action

    assert a.update(obs, 0, True) is None


def test_agents_attackers_round_robin():
    a = RoundRobinAttacker()
    for size in np.random.randint(1, 99, size=4):
        obs = np.full(size, 1)
        zeros = np.random.randint(size, size=size // 3)
        obs[zeros] = 0
        last_action = a.act(obs) - 1
        max_action = max(np.flatnonzero(obs))
        for _ in range(size + size):
            action = a.act(obs) - 1
            assert (last_action < action or last_action == max_action) and action not in zeros
            last_action = action

    assert a.update(obs, 0, True) is None
