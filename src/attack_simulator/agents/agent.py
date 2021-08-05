from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def __init__(self, agent_config=None):
        ...  # pragma: no cover

    @abstractmethod
    def act(self, observation):
        ...  # pragma: no cover

    def update(self, new_observation, reward, done):
        pass

    @property
    def trainable(self):
        return False
