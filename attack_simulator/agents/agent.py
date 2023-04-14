from abc import ABC, abstractmethod
from typing import Any, Dict

from ..constants import UINT


class Agent(ABC):
    """Base class for agents that operate in the simulator."""

    @abstractmethod
    def __init__(self, agent_config: dict) -> None:
        self.num_special_actions = agent_config["num_special_actions"]
        self.wait_action = agent_config["wait_action"]
        self.terminate_action = agent_config["terminate_action"]

    @abstractmethod
    def compute_action_from_dict(self, observation: Dict[str, Any]) -> UINT:
        ...  # pragma: no cover

    @property
    def trainable(self) -> bool:
        return False
