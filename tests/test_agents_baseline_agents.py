import numpy as np
import pytest

from attack_simulator.agents import DisableProbabilityAgent, RandomAgent, RuleBasedAgent, SkipAgent

from .test_tests_utils import np_bits

NUM_TRIALS = 1000


def test_agents_baseline_random():
    num_actions = 3
    visited = np.zeros(num_actions)

    agent = RandomAgent(dict(num_actions=num_actions))

    for _ in range(NUM_TRIALS):
        action = agent.act()
        assert 0 <= action < num_actions
        visited[action] += 1

    assert np.allclose(visited / NUM_TRIALS, 1 / num_actions, atol=0.1)


def test_agents_baseline_skip():
    agent = SkipAgent()

    for _ in range(NUM_TRIALS):
        action = agent.act()
        assert action == 0


def test_agents_baseline_disable_probability(test_graph):
    num_services = test_graph.num_services
    num_attacks = test_graph.num_attacks
    dim_observations = num_services + num_attacks

    num_actions = num_services + 1
    num_options = 1 << num_services
    num_rounds = round(0.5 + NUM_TRIALS / (num_options - 1))

    for p in (0.0, np.random.uniform(0.05, 0.3), 1.0):

        agent = DisableProbabilityAgent(
            dict(disable_probability=p, num_actions=num_services + 1, random_seed=42)
        )

        visited = np.zeros(num_actions)

        for _ in range(1, num_rounds):
            for i in range(1, num_options):
                obs = np_bits(i, size=dim_observations)
                action = agent.act(obs)
                visited[action] += 1
                action -= 1
                assert action == -1 or action in np.flatnonzero(obs[:num_services])

        expected = np.full(num_actions, p / num_services)
        expected[0] = 1 - p
        assert np.allclose(visited / num_rounds / (num_options - 1), expected, atol=0.1)


@pytest.mark.skip(reason="Not working with new conditions. Need to think about rules.")
def test_agents_baseline_rule_based(test_graph):
    agent = RuleBasedAgent(dict(attack_graph=test_graph))
    num_services = test_graph.num_services
    num_attacks = test_graph.num_attacks
    dim_observations = num_services + num_attacks

    for i in range(1, 1 << num_services):
        obs = np_bits(i, size=dim_observations)
        # for each service combination pretend to compromise attack steps one-by-one
        for j in range(num_attacks):
            obs[num_services + j] = 1
            action = agent.act(obs) - 1
            assert action == -1 or action in np.flatnonzero(obs[:num_services])


@pytest.mark.parametrize(
    "agent_class", [RandomAgent, SkipAgent, DisableProbabilityAgent, RuleBasedAgent]
)
def test_not_trainable(agent_class, test_graph):
    agent = agent_class(
        dict(disable_probability=0.5, num_actions=13, random_seed=42, attack_graph=test_graph)
    )
    assert not agent.trainable
