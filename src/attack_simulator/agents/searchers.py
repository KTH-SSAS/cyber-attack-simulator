from collections import deque
from typing import Deque, List, Set, Union

import numpy as np
from numpy.typing import NDArray

from attack_simulator.constants import ACTION_TERMINATE, special_actions

from .agent import Agent

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
        self.rng = np.random.default_rng(agent_config["seed"])

    def act(self, observation: NDArray) -> int:
        surface_indexes = set(np.flatnonzero(observation))
        new_targets = [idx for idx in surface_indexes if idx not in self.targets]

        # Add new targets to the back of the queue
        self.rng.shuffle(new_targets)
        for c in new_targets:
            self.targets.appendleft(c)

        self.current_target = select_next_target(self.current_target, self.targets, surface_indexes)

        if self.current_target == STOP:
            return ACTION_TERMINATE

        # Offset the action by the number of special actions
        return self.current_target + len(special_actions)


class DepthFirstAttacker(Agent):
    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        self.current_target = -1
        self.targets: List[int] = []
        self.rng = np.random.default_rng(agent_config["random_seed"])

    def act(self, observation: NDArray[np.int8]) -> int:
        surface_indexes = set(np.flatnonzero(observation))
        new_targets = [idx for idx in surface_indexes if idx not in self.targets]

        # Add new targets to the top of the stack
        self.rng.shuffle(new_targets)
        for c in new_targets:
            self.targets.append(c)

        if self.current_target in surface_indexes:
            return self.current_target

        self.current_target = select_next_target(self.current_target, self.targets, surface_indexes)

        if self.current_target == STOP:
            return ACTION_TERMINATE

        # Offset the action by the number of special actions
        return self.current_target + len(special_actions)
