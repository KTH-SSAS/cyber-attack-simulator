from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
from ..constants import UINT


class Agent(ABC):
    """Base class for agents that operate in the simulator."""

    @abstractmethod
    def __init__(self, agent_config: dict) -> None:
        pass

    @abstractmethod
    def compute_action_from_dict(self, observation: Dict[str, Any]) -> tuple:
        ...  # pragma: no cover


class RandomActiveAgent(Agent):
    """Agent that will pick a random object to act on each turn."""

    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        seed = agent_config["seed"] if "seed" in agent_config else np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(seed)

    def compute_action_from_dict(self, observation: Dict[str, Any]) -> tuple:
        available_objects = observation["action_mask"][1]
        object_indexes = np.flatnonzero(available_objects)
        return (1, self.rng.choice(object_indexes)) if len(object_indexes) > 0 else (0, 0)


class RandomAgent(Agent):
    """Agent that will pick a random (valid) action each turn."""

    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        seed = agent_config["seed"] if "seed" in agent_config else np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(seed)

    def compute_action_from_dict(self, observation: Dict[str, Any]) -> tuple:
        available_objects = observation["action_mask"][1]
        available_actions = observation["action_mask"][0]
        object_indexes = np.flatnonzero(available_objects)
        action_indexes = np.flatnonzero(available_actions)
        return (
            self.rng.choice(action_indexes) if len(object_indexes) > 0 else 0,
            self.rng.choice(object_indexes) if len(object_indexes) > 0 else 0,
        )


class NothingAgent(Agent):
    """Agent that will always do nothing"""

    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)

    def compute_action_from_dict(self, observation: Dict[str, Any]) -> tuple:
        return (0, 0)
