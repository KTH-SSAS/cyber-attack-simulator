from itertools import tee
from operator import itemgetter
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
from agraphlib import STEP
from networkx.algorithms.shortest_paths.generic import shortest_path
from numpy.typing import NDArray

from attack_simulator.constants import UINT
from attack_simulator.graph import AttackGraph

from .agent import Agent


class PathFinderAttacker(Agent):
    """Attacker with full information of the system.

    Will decide on a series of flags to pursue, and will calculate the
    shortest paths to them.
    """

    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        # graph : AttackGraph = simulator.g #agent_config["attack_graph"]

        self.g: AttackGraph = agent_config["attack_graph"]
        self.flags_taken: List[int] = []
        # Create a random order in which to do flags

        self.prev_defense_state = np.ones(self.g.num_defenses, dtype=bool)
        self.attack_graph: nx.DiGraph = self.g.to_networkx(
            indices=True, system_state=self.prev_defense_state
        )

        self.done = False
        self.flags = []
        self.planned_path: List[int] = []
        self.current_flag_index = 0
        self.current_attack_target = None

        # self.sim.rng.shuffle(self.flags)

        # Decide on a target
        # self.done, planned_path = self.decide_next_target()

        # Path to use as a stack

        # Cover the case of the attacker not being able to do anything the first iteration
        # if not self.done:
        #     self.current_attack_target = self.planned_path.pop()

    def atk_step_idx(self, step_name: Union[str, UINT]) -> UINT:
        if isinstance(step_name, np.intp):
            return step_name

        return self.g.attack_indices[step_name]

    def atk_step_compromised(self, step: Union[int, str], attack_state: NDArray[np.int8]) -> bool:
        if isinstance(step, str):
            step = self.atk_step_idx(step)
        return attack_state[step]

    def skip_steps(self, compromised_steps, path) -> Tuple[int, list]:
        """Skip over steps in path that are already taken."""
        attack_target = path.pop()
        while attack_target in compromised_steps and path:
            attack_target = path.pop()

        return attack_target, path

    def decide_next_target(
        self, ttc_remaining, compromised_steps, defense_state
    ) -> Tuple[bool, List[int]]:
        """Select the next flag to target."""
        path_found = False
        done = True
        path: List[int] = []

        while (not path_found) and self.current_flag_index < len(self.flags) - 1:
            self.current_flag_index += 1
            path_found, path, _ = self.find_path_to(
                self.current_flag_target, ttc_remaining, compromised_steps, defense_state
            )

        # If we ultimately did not find any path, we are done
        done = not path_found

        return done, path

    @property
    def current_flag_target(self) -> int:
        return self.flags[self.current_flag_index]

    def find_path_to(
        self,
        target: UINT,
        ttc_remaining: NDArray[UINT],
        compromised_steps,
        defense_state: NDArray[np.int8],
    ) -> Tuple[bool, List[int], float]:
        """Find a path to an attack step, with respect to AND-steps."""
        total_path: list = []
        # Don't bother pathfinding if the target is defended
        if self.g.step_is_defended(target, defense_state):
            return False, [], 0.0

        try:
            path, _ = self._find_path_to(target, total_path, compromised_steps, ttc_remaining)
        except nx.NodeNotFound:
            return False, [], 0.0
        except nx.NetworkXNoPath:
            return False, [], 0.0

        for step in path:
            _add_unique(total_path, step)

        # The planned path is the steps we actually need to
        # take to reach the target.
        planned_path = list(reversed(total_path))

        planned_path = list(filter(lambda step: step not in compromised_steps, planned_path))
        path_cost = sum(ttc_remaining[n] for n in planned_path)

        return len(total_path) != 0, planned_path, path_cost

    def _validate_path(
        self, path: List[int], total_path: List[int], compromised_steps, ttc_remaining
    ) -> Tuple[List[int], float]:
        ttc_cost = 0.0
        # Check that each step in the path is reacable with respect to AND-steps.
        for node_id in path:
            step = self.attack_graph.nodes[node_id]
            # If node is AND step, go to parents first.
            if step["step_type"] == STEP.AND and node_id not in total_path:
                paths_to_parents = [
                    self._find_path_to(parent, total_path, compromised_steps, ttc_remaining)
                    for parent in self.attack_graph.nodes[node_id]["parents"]
                    if parent not in total_path
                ]
                ttc_cost += _add_paths_to_total_path(paths_to_parents, total_path, ttc_remaining)

            ttc_cost += ttc_remaining[node_id]
            _add_unique(total_path, node_id)

        return path, ttc_cost

    def nx_shortest_path(self, source, target, ttc_remaining):
        # Find the shortest path to the target
        try:
            path_to_target: list = shortest_path(
                self.attack_graph, source=source, target=target, weight="ttc"
            )
            path_cost = sum(ttc_remaining[n] for n in path_to_target)
        except nx.NetworkXNoPath:
            return [], 0.0
        return path_to_target, path_cost

    def _find_path_to(
        self, target: int, total_path: List[int], compromised_steps, ttc_remaining
    ) -> Tuple[List[int], float]:
        """Look for paths from steps in the attack surface."""

        paths = (self.nx_shortest_path(start, target, ttc_remaining) for start in compromised_steps)

        # Go through the available paths in order of decreasing cost
        # and select the first viable one.
        for path_to_target, _ in sorted(paths, key=itemgetter(1)):

            if not path_to_target:
                continue

            try:
                return self._validate_path(
                    path_to_target, total_path, compromised_steps, ttc_remaining
                )
            except nx.NetworkXNoPath:
                continue

        # None of the paths worked
        raise nx.NetworkXNoPath

    def get_attack_target(self, ttc_remaining, attack_surface, compromised_steps, defense_state):
        attack_target = self.current_attack_target

        # Check that the action we chose can be done. Otherwise, select a new path
        if attack_target not in attack_surface:
            flag_taken = self.current_flag_target in compromised_steps
            if flag_taken:
                # If so, select the next target
                self.flags_taken.append(self.current_flag_target)
                self.done, self.planned_path = self.decide_next_target(
                    ttc_remaining, compromised_steps, defense_state
                )
                # If there are no more flags to take we are done
                if self.done:
                    return self.terminate_action

            path_found, self.planned_path, _ = self.find_path_to(
                self.current_flag_target, ttc_remaining, compromised_steps, defense_state
            )
            # If a new path could not be found to the target, target the next flag
            if not path_found:
                self.done, self.planned_path = self.decide_next_target(
                    ttc_remaining, attack_surface, defense_state
                )
                # If there are no more flags to take we are done
                if self.done:
                    return self.terminate_action
            target, self.planned_path = self.skip_steps(compromised_steps, self.planned_path)
            return target

        return attack_target

    def compute_action_from_dict(self, observation: Dict[str, Any]) -> UINT:
        """Follow the current planned path.

        If it's no longer possible to reach the targeted node try to
        recalculate a path. If no path can be found, select a new
        target.
        """

        attack_surface = np.flatnonzero(observation["attack_surface"])
        defense_state = observation["defense_state"]
        ttc_remaining = observation["ttc_remaining"]
        compromised_steps = np.flatnonzero(ttc_remaining == 0)

        if len(self.flags) == 0:
            flag_costs = [
                (flag, self.find_path_to(flag, ttc_remaining, compromised_steps, defense_state)[2])
                for flag in self.g.flag_indices
            ]
            flag_costs.sort(key=itemgetter(1))
            self.flags = [flag for flag, _ in flag_costs]
            # self.done, self.planned_path = self.decide_next_target(ttc_remaining, compromised_steps, defense_state)

        # If there are no more flags to take, do nothing
        if self.done:
            return self.terminate_action

        # If the defender has enabled a defense step, recreate the internal attack graph
        update_graph = not all(self.prev_defense_state == defense_state)
        if update_graph:
            self.attack_graph = self.g.to_networkx(indices=True, system_state=defense_state)
            self.prev_defense_state = np.array(defense_state)

        attack_target = self.get_attack_target(
            ttc_remaining, attack_surface, compromised_steps, defense_state
        )

        assert (
            attack_target in attack_surface
        ), "Attacker tried to perform an attack not in attack surface"

        self.current_attack_target = attack_target
        return attack_target + self.num_special_actions


def _add_unique(path: list, item: Any) -> int:
    if item not in path:
        path.append(item)
        return True

    return False


def _add_paths_to_total_path(
    paths: List[Tuple[List[int], float]], total_path: List[int], ttc_remaining
) -> float:
    added_ttc_cost = 0.0

    for path, ttc in sorted(paths, key=itemgetter(1)):
        steps_not_already_in_path = tee(filter(lambda x: x not in total_path, path))
        added_ttc_cost += sum(ttc_remaining[step] for step in steps_not_already_in_path[0])
        total_path.extend(steps_not_already_in_path[1])

    return added_ttc_cost
