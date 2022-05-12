from abc import ABC, abstractmethod

from numpy import ndarray


class Agent(ABC):
    """Base class for agents that operate in the simulator."""

    @abstractmethod
    def __init__(self, agent_config: dict) -> None:
        self.done = False
        ...  # pragma: no cover

    @abstractmethod
    def act(self, observation: ndarray) -> int:
        ...  # pragma: no cover

    def update(self, new_observation: ndarray, reward: float, done: bool) -> None:
        pass

    @property
    def trainable(self) -> bool:
        return False
