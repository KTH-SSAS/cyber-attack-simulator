from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Base class for agents that operate in the simulator
    """

    @abstractmethod
    def __init__(self, agent_config=None):
        self.done = False
        ...  # pragma: no cover

    @abstractmethod
    def act(self, observation):
        ...  # pragma: no cover

    def update(self, new_observation, reward, done):
        pass

    @property
    def trainable(self):
        return False
