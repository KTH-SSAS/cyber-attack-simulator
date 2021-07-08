from abc import ABC, abstractmethod
import numpy as np
from numpy.core.fromnumeric import argmax
import numpy.random as random

class Agent(ABC):

    @abstractmethod
    def act(self, observation):
        ...

    @abstractmethod
    def update(self, action, reward):
        ...


class BanditAgent(Agent):

    def __init__(self, num_actions) -> None:
        self.Q = np.array(num_actions)
        self.times_taken = np.array(num_actions)

    def update(self, action, reward):
        self.times_taken[action] += 1
        self.Q[action] = self.Q[action] + \
            (reward-self.Q[action])/self.times_taken[action]

    def act(self, observation):
        return np.argmax(self.Q)


class QLearningAgent(Agent):

    def __init__(self, num_actions) -> None:

        self.num_actions = num_actions
        self.Q = {}  # A dict is used to save on memory
        self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 0.5
        self.state = None
        self.action = None

    def act(self, state):
        self.state = state
        if random.rand() > self.epsilon:
            a = self.Q.get(state)
            if a is None:
                a = [10]*self.num_actions
                self.Q[state] = a
            action = argmax(a)
        else:
            action = random.randint(self.num_actions)

        self.action = action
        return action

    def update(self, new_state, reward):
        q_prime = self.Q.get(new_state)
        if q_prime is None:
            q_prime = [10]*self.num_actions
            self.Q[new_state] = q_prime

        q = self.Q[self.state][self.action]
        self.Q[self.state][self.action] = q + self.alpha * \
            (reward + self.gamma*argmax(q_prime) - q)


class RandomAgent(Agent):

    def __init__(self, disable_probability):
        self.disable_probability = disable_probability

    def act(self, enabled_services):
        for service in enabled_services:
            if enabled_services[service] == 1 and random.uniform(0, 1) < self.disable_probability:
                enabled_services[service] = 0
        return None

    def update(self, action, reward):
        return
