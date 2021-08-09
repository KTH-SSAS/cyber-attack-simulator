import numpy as np
import pytest

from attack_simulator.agents.tabular_agents import BanditAgent, QLearningAgent, epsilon_greedy

from .test_agents_policy_agents import _run_episodes

NUM_TRIALS = 1000


def test_agents_tabular_epsilon_greedy():
    num_actions = 3
    values = np.zeros(num_actions)
    values[1] = 1

    for epsilon in (0.0, np.random.uniform(0.05, 0.3), 1.0):
        visited = np.zeros(num_actions)
        for _ in range(NUM_TRIALS):
            visited[epsilon_greedy(values, epsilon)] += 1

        p = epsilon / num_actions
        expected = np.full(num_actions, p)
        expected[1] += 1 - epsilon
        assert np.allclose(visited / NUM_TRIALS, expected, atol=0.1)


def test_agents_tabular_bandit():
    num_actions = 3
    visited = np.zeros(num_actions)

    agent = BanditAgent(dict(num_actions=num_actions, random_seed=42))

    for _ in range(NUM_TRIALS):
        action = agent.act()
        assert 0 <= action < num_actions
        visited[action] += 1
        agent.update(None, np.random.uniform(0, 1), False)

    assert all(visited)


def test_agents_tabular_qlearning(test_env):
    agent = QLearningAgent(dict(num_actions=test_env.action_space.n, random_seed=42))

    # train
    _, train_mean = _run_episodes(NUM_TRIALS, test_env, agent, True)

    # evaluate
    _, eval_mean = _run_episodes(NUM_TRIALS, test_env, agent)

    # did we learn anything?
    assert train_mean <= eval_mean


@pytest.mark.parametrize("agent_class", [BanditAgent, QLearningAgent])
def test_trainable(agent_class, test_env):
    agent = agent_class(dict(num_actions=11, random_seed=42))
    assert agent.trainable
