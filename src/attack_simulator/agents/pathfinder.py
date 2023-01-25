from operator import itemgetter
from typing import Any, List, Tuple, Union

import networkx as nx
import numpy as np
from agraphlib import STEP
from networkx.algorithms.shortest_paths.generic import shortest_path

from attack_simulator.sim import AttackSimulator

from .agent import Agent

DO_NOTHING = -1


class PathFinderAttacker(Agent):
    """Attacker with full information of the system.

    Will decide on a series of flags to pursue, and will calculate the
    shortest paths to them.
    """

    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        self.sim: AttackSimulator = agent_config["simulator"]
        # graph : AttackGraph = simulator.g #agent_config["attack_graph"]

        self.flags_taken: List[int] = []
        # Create a random order in which to do flags

        self.defense_state = np.array(self.sim.defense_state)
        self.attack_graph: nx.DiGraph = self.sim.g.to_networkx(
            indices=True, system_state=self.sim.defense_state
        )

        self.start_node = self.sim.entry_attack_index

        self.flags = list(self.sim.g.flag_indices)

        flag_costs = [(flag, self.find_path_to(flag)[2]) for flag in self.flags]

        flag_costs.sort(key=itemgetter(1))

        self.flags = [flag for flag, _ in flag_costs]

        # self.sim.rng.shuffle(self.flags)

        # Decide on a target
        self.current_flag_index = -1
        self.done, planned_path = self.decide_next_target()

        # Path to use as a stack
        self.planned_path: List[int] = planned_path

        # Cover the case of the attacker not being able to do anything the first iteration
        if not self.done:
            self.current_attack_target = self.planned_path.pop()

    def atk_step_idx(self, step_name: Union[str, int]) -> int:
        if isinstance(step_name, int):
            return step_name

        return self.sim.g.attack_indices[step_name]

    def atk_step_compromised(self, step: Union[int, str]) -> bool:
        if isinstance(step, str):
            step = self.atk_step_idx(step)
        return bool(self.sim.attack_state[step])

    def skip_steps(self) -> int:
        """Skip over steps in path that are already taken."""
        attack_target = self.planned_path.pop()
        while self.atk_step_compromised(attack_target) and self.planned_path:
            attack_target = self.planned_path.pop()

        return attack_target

    def decide_next_target(self) -> Tuple[bool, List[int]]:
        """Select the next flag to target."""
        path_found = False
        done = True
        path: List[int] = []

        while (not path_found) and self.current_flag_index < len(self.flags) - 1:
            self.current_flag_index += 1
            path_found, path, _ = self.find_path_to(self.current_flag_target)

        # If we ultimately did not find any path, we are done
        done = not path_found

        return done, path

    @property
    def current_flag_target(self) -> int:
        return self.flags[self.current_flag_index]

    def find_path_to(self, target: int) -> Tuple[bool, List[int], float]:
        """Find a path to an attack step, with respect to AND-steps."""
        total_path: list = []
        # Don't bother pathfinding if the target is defended
        if self.sim.g.step_is_defended(target, self.defense_state):
            return False, [], 0.0

        try:
            path, _ = self._find_path_to(target, total_path)
        except nx.NodeNotFound:
            return False, [], 0.0
        except nx.NetworkXNoPath:
            return False, [], 0.0

        for step in path:
            _add_unique(total_path, step)

        # The planned path is the steps we actually need to
        # take to reach the target.
        planned_path = list(reversed(total_path))

        planned_path = [step for step in planned_path if not self.atk_step_compromised(step)]
        path_cost = sum(self.sim.ttc_remaining[n] for n in planned_path)

        return len(total_path) != 0, planned_path, path_cost

    def _validate_path(self, path: List[int], total_path: List[int]) -> Tuple[List[int], float]:
        ttc_cost = 0.0
        # Check that each step in the path is reacable with respect to AND-steps.
        for node_id in path:
            step = self.attack_graph.nodes[node_id]
            # If node is AND step, go to parents first.
            if step["step_type"] == STEP.AND and node_id not in total_path:
                parents = self.attack_graph.nodes[node_id]["parents"]
                paths_to_parents = []
                for parent in parents:
                    # If the parent is already in the path, there is no need to find a path to it
                    if parent not in total_path:
                        paths_to_parents.append(self._find_path_to(parent, total_path))

                    ttc_cost += _add_paths_to_total_path(paths_to_parents, total_path)

            ttc_cost += self.sim.ttc_remaining[node_id]
            _add_unique(total_path, node_id)

        return path, ttc_cost

    def _find_path_to(self, target: int, total_path: List[int]) -> Tuple[List[int], float]:
        """Look for paths from steps in the attack surface."""
        paths = []
        surface = self.attack_surface
        for initial_step in surface:
            try:
                # Find the shortest path to the target
                path_to_target: list = shortest_path(
                    self.attack_graph, source=initial_step, target=target, weight="ttc"
                )
                path_cost = sum([self.sim.ttc_remaining[n] for n in path_to_target])
                paths.append((path_to_target, path_cost))
            except nx.NetworkXNoPath:
                continue

        # Go through the available paths in order of decreasing cost
        # and select the first viable one.
        for path_to_target, _ in sorted(paths, key=itemgetter(1)):
            try:
                return self._validate_path(path_to_target, total_path)
            except nx.NetworkXNoPath:
                continue

        # None of the paths worked
        raise nx.NetworkXNoPath

    @property
    def attack_surface(self) -> np.ndarray:
        """Returns all attack steps that are possible to attack."""
        return self.sim.valid_actions

    def act(self, observation: np.ndarray) -> int:
        """Follow the current planned path.

        If it's no longer possible to reach the targeted node try to
        recalculate a path. If no path can be found, select a new
        target.
        """

        attack_surface = observation[0]

        # If there are no more flags to take, do nothing
        if self.done:
            return DO_NOTHING

        # If the defender has enabled a defense step, recreate the internal attack graph
        update_graph = not all(self.defense_state == self.sim.defense_state)
        if update_graph:
            self.attack_graph = self.sim.g.to_networkx(
                indices=True, system_state=self.sim.defense_state
            )
            self.defense_state = np.array(self.sim.defense_state)

        attack_target = self.current_attack_target
        current_step_is_compromised = self.atk_step_compromised(attack_target)

        if current_step_is_compromised:
            # Check if we have taken the current flag target
            flag_taken = attack_target == self.current_flag_target
            if flag_taken:
                # If so, select the next target
                self.flags_taken.append(self.current_flag_target)
                self.done, self.planned_path = self.decide_next_target()
                # If there are no more flags to take we are done
                if self.done:
                    return DO_NOTHING

            # Select the next attack step to work on
            attack_target = self.skip_steps()

        # Check that the action we chose can be done. Otherwise, select a new path
        attack_step_not_available = not attack_surface[attack_target]
        if attack_step_not_available:
            path_found, self.planned_path, _ = self.find_path_to(self.current_flag_target)
            # If a new path could not be found to the target, target the next flag
            if not path_found:
                self.done, self.planned_path = self.decide_next_target()
                # If there are no more flags to take we are done
                if self.done:
                    return DO_NOTHING
            attack_target = self.skip_steps()

        assert (
            attack_target in self.sim.valid_actions
        ), "Attacker tried to perform an attack not in attack surface"

        self.current_attack_target = attack_target
        return attack_target


def _add_unique(path: list, item: Any) -> int:
    if item not in path:
        path.append(item)
        return True

    return False


def _add_paths_to_total_path(paths: List[Tuple[List[int], float]], total_path: List[int]) -> float:
    added_ttc_cost = 0.0
    for path, ttc in sorted(paths, key=itemgetter(1)):
        for node in path:
            added = _add_unique(total_path, node)
            if added:
                added_ttc_cost += ttc
    return added_ttc_cost
