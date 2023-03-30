from collections import deque
from typing import Any, Deque, Dict, List, Set, Union

import numpy as np

from ... import UINT
from ..agent import Agent

STOP = -1


def get_new_targets(attack_surface: Set[int], discovered_targets: Set[int]) -> List[int]:
    new_targets = attack_surface.difference(discovered_targets)
    return list(sorted(new_targets))


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


class BreadthFirstAttacker(Agent):
    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        self.current_target = STOP
        self.targets: Deque[int] = deque([])

        seed = agent_config["seed"] if "seed" in agent_config else np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(seed)

    def compute_action_from_dict(self, observation: Dict[str, Any]) -> UINT:
        attack_surface = observation["action_mask"].reshape(-1)[self.num_special_actions :]
        surface_indexes = set(np.flatnonzero(attack_surface))
        new_targets = [idx for idx in surface_indexes if idx not in self.targets]

        # Add new targets to the back of the queue
        self.rng.shuffle(new_targets)
        for c in new_targets:
            self.targets.appendleft(c)

        self.current_target = select_next_target(self.current_target, self.targets, surface_indexes)

        if self.current_target == STOP:
            return self.terminate_action

        # Offset the action by the number of special actions
        return self.current_target + self.num_special_actions


class DepthFirstAttacker(Agent):
    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        self.current_target = -1
        self.targets: List[int] = []
        seed = agent_config["seed"] if "seed" in agent_config else np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(seed)

    def compute_action_from_dict(self, observation: Dict[str, Any]) -> UINT:
        attack_surface = observation["action_mask"].reshape(-1)[self.num_special_actions :]
        surface_indexes = set(np.flatnonzero(attack_surface))
        new_targets = [idx for idx in surface_indexes if idx not in self.targets]

        # Add new targets to the top of the stack
        self.rng.shuffle(new_targets)
        for c in new_targets:
            self.targets.append(c)

        self.current_target = select_next_target(self.current_target, self.targets, surface_indexes)

        if self.current_target == STOP:
            return self.terminate_action

        # Offset the action by the number of special actions
        return self.current_target + self.num_special_actions
