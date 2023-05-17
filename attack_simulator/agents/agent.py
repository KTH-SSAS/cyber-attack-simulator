from abc import ABC, abstractmethod
from typing import Any, Dict

from .. import UINT


class Agent(ABC):
    """Base class for agents that operate in the simulator."""

    @abstractmethod
    def __init__(self, agent_config: dict) -> None:
        pass

    @abstractmethod
    def compute_action_from_dict(self, observation: Dict[str, Any]) -> UINT:
        ...  # pragma: no cover

    @property
    def trainable(self) -> bool:
        return False
