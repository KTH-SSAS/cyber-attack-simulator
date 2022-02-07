"""
This module implements Attacker agents

Note that the observation space for attackers is the current attack surface,
and their action space is 0 for "no action" or 1 + the index of an attack in
the current attack surface; essentially [0, num_attacks] inclusive.
"""
import logging
from operator import itemgetter
from typing import List

import networkx as nx
import numpy as np
from networkx.algorithms.shortest_paths.generic import shortest_path

from attack_simulator.sim import AttackSimulator

from ..rng import get_rng
from .agent import Agent

logger = logging.getLogger("simulator")


class RandomAttacker(Agent):
    def __init__(self, agent_config):
        super().__init__()
        self.rng, _ = get_rng(agent_config.get("random_seed"))

    def act(self, observation):
        valid_attack_indices = np.flatnonzero(observation)
        return self.rng.choice(valid_attack_indices) + 1


class RandomNoActionAttacker(Agent):
    def __init__(self, agent_config):
        super().__init__()
        self.rng, _ = get_rng(agent_config.get("random_seed"))

    def act(self, observation):
        valid_attacks = np.concatenate([[0], np.flatnonzero(observation) + 1])
        return self.rng.choice(valid_attacks)


class RoundRobinAttacker(Agent):
    def __init__(self, agent_config=None):
        super().__init__()
        self.last = 0

    def act(self, observation):
        valid = np.flatnonzero(observation)
        above = valid[self.last < valid]
        self.last = valid[0] if 0 == above.size else above[0]
        return self.last + 1


class RoundRobinNoActionAttacker(Agent):
    def __init__(self, agent_config=None):
        super().__init__()
        self.last = 0

    def act(self, observation):
        valid = np.concatenate([[0], np.flatnonzero(observation) + 1])
        above = valid[self.last < valid]
        self.last = valid[0] if 0 == above.size else above[0]
        return self.last


class WellInformedAttacker(Agent):
    """An Attacker with complete information on the underlying AttackGraph"""

    def __init__(self, agent_config):
        super().__init__()
        graph = agent_config["attack_graph"]
        steps = graph.attack_steps
        names = graph.attack_names
        self._ttc = dict(zip(names, agent_config["ttc"]))
        self._rewards = dict(zip(names, agent_config["rewards"]))
        values = {}
        """
        self._value = (
        lambda x, y, z: (_ for _ in ()).throw(RuntimeError("called disabled method"))
        )
        """
        total = self._value(steps, values, graph.root)
        logger.info("%s: total discounted value: %d", self.__class__.__name__, total)
        del self._ttc
        del self._rewards

        self.attack_values = np.array([values[name] for name in names])

    def _value(self, attack_steps, attack_values, attack_name, discount_rate=0.1):
        """
        Recursively compute the value of each attack step.

        The discount rate is meant to account for uncertainity in future rewards
        due the defender's actions possibly disabling relevant services.

        Note: Does not consider AND steps, so will not always act optimally.
        """
        if attack_name not in attack_values:
            attack_step = attack_steps[attack_name]
            value = self._rewards[attack_name]
            for child_name in attack_step.children:
                value += self._value(attack_steps, attack_values, child_name)
            value /= (1 + discount_rate) ** self._ttc[attack_name]
            attack_values[attack_name] = value
        return attack_values[attack_name]

    def act(self, observation):
        """Selecting the attack step with the highest net present value."""
        return np.argmax(self.attack_values * observation) + 1


class InformedAttacker(WellInformedAttacker):
    """An Attacker with access to the AttackGraph **without sampled TTC and rewards**"""

    def __init__(self, agent_config):
        graph = agent_config["attack_graph"]
        steps = graph.attack_steps
        names = graph.attack_names
        # replace per-episode sampled info with base parameters
        agent_config["ttc"] = np.array([steps[name].ttc for name in names])
        agent_config["rewards"] = np.array([steps[name].reward for name in names])
        super(InformedAttacker, self).__init__(agent_config)


class PathFinderAttacker:
    """
    Attacker with full information of the system.
    Will decide on a series of flags to pursue,
    and will calculate the shortest paths to them.
    """

    def __init__(self, agent_config) -> None:

        self.sim: AttackSimulator = agent_config["simulator"]
        # graph : AttackGraph = simulator.g #agent_config["attack_graph"]

        self.attack_surface = [self.sim.g.attack_names[idx] for idx in self.sim.valid_actions]

        self.flags = [
            self.sim.g.attack_names[idx] for idx in np.nonzero(self.sim.g.reward_params)[0]
        ]
        self.flags_taken: List[str] = []
        # Create a random order in which to do flags
        self.sim.rng.shuffle(self.flags)

        self.service_states = np.array(self.sim.service_state)
        self.attack_graph: nx.DiGraph = self.sim.g.to_networkx(
            indices=False, system_state=self.sim.service_state
        )
        self.start_node = self.sim.g.attack_names[
            self.sim.entry_attack_index
        ]  # self.sim.entry_attack_index #agent_config["initial_step"]
        self.total_path: List[int] = []

        self.done = False

        # Decide on a target
        self.current_flag_index = 0
        self.current_flag_target = self.flags[self.current_flag_index]

        # Path to use as a stack
        self.planned_path: List[str] = []

        self.find_path_to(self.current_flag_target)

        self.current_attack_target = self.planned_path.pop()

    def skip_steps(self):
        """
        Skip over steps in path that are already taken
        """
        attack_target = self.planned_path.pop()
        attack_index = self.sim.g.attack_indices[attack_target]
        while self.sim.attack_state[attack_index]:
            attack_target = self.planned_path.pop()
            attack_index = self.sim.g.attack_indices[attack_target]

        return attack_target, attack_index

    def decide_next_target(self):
        """
        Select the next flag to target
        """
        path_found = False
        done = True

        while (not path_found) and self.current_flag_index < len(self.flags) - 1:
            self.current_flag_index += 1
            self.current_flag_target = self.flags[self.current_flag_index]
            path_found = self.find_path_to(self.current_flag_target)

        # If we ultimately did not find any path, we are done
        done = not path_found

        return done

    def find_path_to(self, target):
        """Find a path to an attack step,
        with respect to AND-steps"""
        self.total_path = []
        # Don't bother pathfinding if the target's conditions are unavailable
        if not self.sim.g._attack_step_reachable(
            self.sim.g.attack_indices[target], self.sim.service_state
        ):
            return False

        try:
            path, _ = self._find_path_to(target)
        except nx.NodeNotFound:
            return False
        except nx.NetworkXNoPath:
            return False

        for step in path:
            _add_unique(self.total_path, step)

        self.planned_path = list(reversed(self.total_path))
        return len(self.total_path) != 0

    def _validate_path(self, path):
        ttc_cost = 0
        # Check that each step in the path is reacable with respect to AND-steps.
        for node_id in path:
            step = self.attack_graph.nodes[node_id]
            # If node is AND step, go to parents first.
            if step["step_type"] == "and" and node_id not in self.total_path:
                parents = self.attack_graph.nodes[node_id]["parents"]
                paths_to_parents = []
                for parent in parents:
                    # If the parent is already in the path, there is no need to find a path to it
                    if parent not in self.total_path:
                        paths_to_parents.append(self._find_path_to(parent))

                    ttc_cost += _add_paths_to_total_path(paths_to_parents, self.total_path)

            ttc_cost += step["ttc"]
            _add_unique(self.total_path, node_id)

        return path, ttc_cost

    def _find_path_to(self, target):
        # Look for paths from every available step in the attack surface.
        paths = []
        for initial_step in self.attack_surface:
            try:
                # Find the shortest path to the target
                path_to_target: list = shortest_path(
                    self.attack_graph, source=initial_step, target=target, weight="ttc"
                )
                path_cost = sum([self.attack_graph.nodes[n]["ttc"] for n in path_to_target])
                paths.append((path_to_target, path_cost))
            except nx.NetworkXNoPath:
                continue

        for path_to_target, _ in sorted(paths, key=itemgetter(1)):
            try:
                return self._validate_path(path_to_target)
            except nx.NetworkXNoPath:
                continue

        # None of the paths worked
        raise nx.NetworkXNoPath

    def act(self, observation):
        """
        Follow the current planned path.
        If it's no longer possible to reach the targeted node
        try to recalculate a path. If no path can be found,
        select a new target.
        """
        self.attack_surface = [self.sim.g.attack_names[idx] for idx in self.sim.valid_actions]

        # If there are no more flags to take, do nothing
        if self.done:
            return 0

        # If the defender has shut something down, recreate the internal attack graph
        update_graph = not all(self.service_states == self.sim.service_state)
        if update_graph:
            self.attack_graph: nx.DiGraph = self.sim.g.to_networkx(
                indices=False, system_state=self.sim.service_state
            )
            self.service_states = np.array(self.sim.service_state)

        attack_target = self.current_attack_target
        attack_index = self.sim.g.attack_indices[attack_target]
        current_step_is_compromised = bool(self.sim.attack_state[attack_index])

        if current_step_is_compromised:
            # Check if we have taken the current flag target
            if attack_target == self.current_flag_target:
                # If so, select the next target
                self.flags_taken.append(self.current_flag_target)
                self.done = self.decide_next_target()
                # If there are no more flags to take we are done
                if self.done:
                    return 0

            # Select the next attack step to work on
            attack_target, attack_index = self.skip_steps()

        # Check that the action we chose can be done. Otherwise, select a new path
        attack_step_not_available = not observation[attack_index]
        if attack_step_not_available:
            path_found = self.find_path_to(self.current_flag_target)
            # If a new path could not be found to the target, target the next flag
            if not path_found:
                self.done = self.decide_next_target()
                # If there are no more flags to take we are done
                if self.done:
                    return 0
            attack_target, attack_index = self.skip_steps()

        dependent_services = self.sim.g.service_index_by_attack_index[attack_index]
        dependent_service_states = self.sim.service_state[dependent_services]
        assert all(dependent_service_states)

        assert (
            self.sim.g.attack_indices[attack_target] in self.sim.valid_actions
        ), "Attacker tried to perform an attack not in attack surface"

        self.current_attack_target = attack_target
        return self.sim.g.attack_indices[attack_target] + 1


def _add_unique(path: list, item):
    if item not in path:
        path.append(item)
        return True
    else:
        return False


def _add_paths_to_total_path(paths, total_path):
    added_ttc_cost = 0
    for path, ttc in sorted(paths, key=itemgetter(1)):
        for node in path:
            added = _add_unique(total_path, node)
            if added:
                added_ttc_cost += ttc
    return added_ttc_cost
