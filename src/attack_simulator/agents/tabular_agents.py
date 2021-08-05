from collections import defaultdict

import numpy as np

from ..rng import get_rng
from .agent import Agent


def _argmax_random_tie_break(values, rng):
    return rng.choice(np.flatnonzero(values == np.max(values)))


def epsilon_greedy(values, epsilon=0.1, rng=None):
    if rng is None:
        rng = np.random

    if rng.uniform(0, 1) < epsilon:
        action = rng.choice(range(len(values)))
    else:
        action = _argmax_random_tie_break(values, rng)

    return action


class BanditAgent(Agent):
    def __init__(self, agent_config):
        self.rng, _ = get_rng(agent_config.get("random_seed"))
        num_actions = agent_config["num_actions"]
        self.Q = np.zeros(num_actions)
        self.times_taken = np.zeros(num_actions, dtype="int")
        self.action = None

    def update(self, new_observation, reward, done):
        action = self.action
        self.times_taken[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.times_taken[action]

    def act(self, observation=None):
        self.action = epsilon_greedy(self.Q, rng=self.rng)
        return self.action

    @property
    def trainable(self):
        return True


class QLearningAgent(Agent):
    def __init__(self, agent_config):
        self.rng, _ = get_rng(agent_config.get("random_seed"))
        self.num_actions = agent_config["num_actions"]
        self.alpha = 0.5
        self.gamma = 0.5
        self.state = None
        self.action = None
        # A defaultdict is used to save on memory and automatically fill missing entries
        # TODO: explain the magic constant 10
        self.Q = defaultdict(lambda: np.full(self.num_actions, 10))

    def act(self, observation):
        self.state = tuple(observation)
        self.action = epsilon_greedy(self.Q[self.state], rng=self.rng)
        return self.action

    def update(self, new_observation, reward, done):
        q_prime = 0 if done else self.Q[tuple(new_observation)]
        q = self.Q[self.state][self.action]
        self.Q[self.state][self.action] += self.alpha * (reward + self.gamma * np.max(q_prime) - q)

    @property
    def trainable(self):
        return True
