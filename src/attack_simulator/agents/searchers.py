from collections import deque
from typing import Deque, List, Set, Union
from numpy.typing import NDArray
import numpy as np
from .agent import Agent


DO_NOTHING = -1


def get_new_targets(attack_surface: Set[int], discovered_targets: Set[int]) -> List[int]:
    new_targets = attack_surface.difference(discovered_targets)
    return list(sorted(new_targets))


def select_next(targets: Union[List[int], Deque[int]]) -> int:

    if len(targets) == 0:
        return DO_NOTHING

    current_target = targets.pop()

    return current_target


def select_next_target(
    current_target: int, targets: Union[List[int], Deque[int]], attack_surface: Set[int]
) -> int:

    if current_target in attack_surface:
        return current_target

    while current_target not in attack_surface:
        current_target = select_next(targets)
        if current_target == DO_NOTHING:
            break
    return current_target


class BreadthFirstAttacker(Agent):
    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        self.current_target = DO_NOTHING
        self.targets: Deque[int] = deque([])
        self.discovered_targets: Set[int] = set()
        self.rng = np.random.default_rng(agent_config["random_seed"])

    def act(self, observation: NDArray) -> int:
        attack_surface = observation[0]
        surface_indexes = set(np.flatnonzero(attack_surface))
        new_targets = get_new_targets(surface_indexes, self.discovered_targets)

        # Add new targets to the back of the queue
        self.rng.shuffle(new_targets)
        for c in new_targets:
            self.targets.appendleft(c)

        self.discovered_targets.update(new_targets)

        self.current_target = select_next_target(self.current_target, self.targets, surface_indexes)

        return self.current_target


class DepthFirstAttacker(Agent):
    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        self.current_target = DO_NOTHING
        self.targets: List[int] = []
        self.discovered_targets: Set[int] = set()
        self.rng = np.random.default_rng(agent_config["random_seed"])

    def act(self, observation: NDArray[np.int8]) -> int:

        attack_surface = observation[0]
        surface_indexes = set(np.flatnonzero(attack_surface))
        new_targets = get_new_targets(surface_indexes, self.discovered_targets)
        self.discovered_targets.update(new_targets)

        # Add new targets to the top of the stack
        self.rng.shuffle(new_targets)
        for c in new_targets:
            self.targets.append(c)

        if self.current_target in surface_indexes:
            return self.current_target

        self.current_target = select_next_target(self.current_target, self.targets, surface_indexes)

        return self.current_target
