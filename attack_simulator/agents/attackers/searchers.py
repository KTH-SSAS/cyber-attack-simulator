from collections import deque
from typing import Any, Deque, Dict, List, Set, Union

import numpy as np

from ... import UINT
from ..agent import Agent


def get_new_targets(attack_surface: Set[int], discovered_targets: Set[int]) -> List[int]:
    new_targets = attack_surface.difference(discovered_targets)
    return list(sorted(new_targets))


class BreadthFirstAttacker(Agent):
    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        self.targets: Deque[int] = deque([])
        self.current_target: int = None
        seed = agent_config["seed"] if "seed" in agent_config else np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(seed)

    def compute_action_from_dict(self, observation: Dict[str, Any]) -> UINT:
        attack_surface = observation["attack_surface"]
        surface_indexes = set(np.flatnonzero(attack_surface))
        new_targets = [idx for idx in surface_indexes if idx not in self.targets]

        # Add new targets to the back of the queue
        self.rng.shuffle(new_targets)
        for c in new_targets:
            self.targets.appendleft(c)

        self.current_target, done = self.select_next_target(
            self.current_target, self.targets, surface_indexes
        )

        action = observation["nop_index"] if done else 1
        self.current_target = None if done else self.current_target

        # Offset the action by the number of special actions
        return (action, self.current_target)

    @staticmethod
    def select_next_target(
        current_target: int, targets: Union[List[int], Deque[int]], attack_surface: Set[int]
    ) -> int:

        # If the current target was not compromised, put it
        # back, but on the bottom of the stack.
        if current_target in attack_surface:
            targets.appendleft(current_target)
            current_target = targets.pop()

        while current_target not in attack_surface:
            if len(targets) == 0:
                return None, True

            current_target = targets.pop()

        return current_target, False


class DepthFirstAttacker(Agent):
    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        self.current_target = -1
        self.targets: List[int] = []
        seed = agent_config["seed"] if "seed" in agent_config else np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(seed)

    def compute_action_from_dict(self, observation: Dict[str, Any]) -> UINT:
        attack_surface = observation["action_mask"].reshape(-1)[observation["action_offset"] :]
        surface_indexes = set(np.flatnonzero(attack_surface))
        new_targets = [idx for idx in surface_indexes if idx not in self.targets]

        # Add new targets to the top of the stack
        self.rng.shuffle(new_targets)
        for c in new_targets:
            self.targets.append(c)

        self.current_target = self.select_next_target(
            self.current_target, self.targets, surface_indexes
        )

        if self.current_target == STOP:
            return observation["nop_index"]

        # Offset the action by the number of special actions
        return self.current_target + observation["action_offset"]

    @staticmethod
    def select_next_target(
        current_target: int, targets: Union[List[int], Deque[int]], attack_surface: Set[int]
    ) -> int:

        if current_target in attack_surface:
            return current_target

        while current_target not in attack_surface:

            if len(targets) == 0:
                return STOP

            current_target = targets.pop()

        return current_target
