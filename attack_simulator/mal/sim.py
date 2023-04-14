import pickle
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import ACTION_TERMINATE, ACTION_WAIT, AGENT_ATTACKER, AGENT_DEFENDER, UINT
from ..utils.config import SimulatorConfig
from ..utils.rng import get_rng
from .graph import AttackGraph
from .ids import ProbabilityIDS
from .observation import Info, Observation


class Simulator(ABC):
    num_special_actions: int
    wait_action: int
    terminate_action: int
    ttc_total: int

    @abstractmethod
    def reset(self) -> Tuple[Observation, Info]:
        pass

    @abstractmethod
    def step(self, actions: Dict[str, int]) -> Tuple[Observation, Info]:
        pass


class SimulatorState:
    def __init__(self, graph: AttackGraph, config: SimulatorConfig) -> None:
        entrypoint = graph.attack_indices[graph.root]
        rng = get_rng(config.seed)[0]
        self.time = 0  # time
        self.service_state = np.ones(graph.num_services, dtype="int8")  # service state
        self.defense_state = np.ones(graph.num_defenses, dtype="int8")  # defense state
        self.attack_state = np.zeros(graph.num_attacks, dtype="int8")  # attack state
        self.false_positives = np.zeros(graph.num_attacks, dtype="int8")  # false positives
        self.false_negatives = np.zeros(graph.num_attacks, dtype="int8")  # false negatives
        self.last_observation = np.zeros(graph.num_attacks, dtype="int8")  # last observation
        self.rng = rng  # rng
        self.noise = rng.random(graph.num_attacks, dtype=np.float64)  # noise
        self.attack_start_time = int(rng.exponential(config.attack_start_time))
        # attack start time
        self.num_observed_alerts = 0  # num observed alerts
        self.prev_actions = {AGENT_ATTACKER: entrypoint, AGENT_DEFENDER: ACTION_WAIT}
        self.valid_actions = {AGENT_ATTACKER: False, AGENT_DEFENDER: False}
        # Set the TTC for the entry attack to be the attack start time
        self.ttc_remaining, self.ttc_total = self.get_initial_ttc_vals(
            graph, self.attack_start_time, config.randomize_ttc
        )
        self.attack_surface = self.get_initial_attack_surface(graph, self.attack_start_time)
        self.current_attack_step = entrypoint

    @property
    def attack_surface_empty(self) -> bool:
        return not any(self.attack_surface)

    @property
    def valid_attacks(self) -> NDArray[np.intp]:
        return np.flatnonzero(self.attack_surface)

    @property
    def valid_defenses(self) -> NDArray[np.intp]:
        return np.flatnonzero(self.defense_state)

    @property
    def compromised_steps(self) -> np.ndarray:
        return np.flatnonzero(self.attack_state)

    def get_vulnerable_children(self, graph: AttackGraph, attack_index: UINT) -> List[UINT]:
        return graph.get_vulnerable_children(attack_index, self.attack_state, self.defense_state)

    def get_initial_attack_surface(self, graph: AttackGraph, attack_start_time: UINT) -> NDArray:
        entrypoint = graph.attack_indices[graph.root]
        attack_surface = np.zeros(graph.num_attacks, dtype="int8")
        if attack_start_time == 0:
            attack_surface[entrypoint] = 0
            self.attack_state[entrypoint] = 1
            # add reachable steps to the attack surface
            attack_surface[self.get_vulnerable_children(graph, entrypoint)] = 1
        else:
            attack_surface[entrypoint] = 1

        return attack_surface

    def get_initial_ttc_vals(
        self, graph: AttackGraph, attack_start_time: UINT, randomize_ttc: bool
    ) -> Tuple[NDArray[UINT], UINT]:
        entrypoint = graph.attack_indices[graph.root]
        if randomize_ttc:
            ttc_remaining = np.array(
                [v if v == 0 else max(1, v) for v in self.rng.exponential(graph.ttc_params)],
                dtype=UINT,
            )
        else:
            ttc_remaining = np.array(graph.ttc_params, dtype=UINT)

        ttc_remaining[entrypoint] = attack_start_time

        return ttc_remaining, np.sum(ttc_remaining)


class AttackSimulator(Simulator):
    """Does the simulation."""

    def __init__(self, config: SimulatorConfig, attack_graph: AttackGraph) -> None:
        ## Constants
        self.config = config
        self.ids = ProbabilityIDS(
            config.seed, config.false_negative_rate, config.false_positive_rate
        )
        self.fnr = config.false_negative_rate
        self.fpr = config.false_positive_rate
        self.g: AttackGraph = attack_graph
        self.entry_attack_index: UINT = self.g.attack_indices[self.g.root]

        # A special action is an action that is not related to a attack or
        # defense step
        self.special_actions = [ACTION_WAIT, ACTION_TERMINATE]
        self.action_indices = {
            ACTION_WAIT: 0,
            ACTION_TERMINATE: 1,
        }
        self.action_strings = {
            0: ACTION_WAIT,
            1: ACTION_TERMINATE,
        }
        self.wait_action = self.action_indices[ACTION_WAIT]
        self.terminate_action = self.action_indices[ACTION_TERMINATE]
        self.num_special_actions = len(self.special_actions)

        ## State
        self.state = SimulatorState(self.g, config)

    @property
    def num_flags(self) -> int:
        return self.g.flag_indices.shape[0]

    @property
    def num_attack_steps(self) -> UINT:
        return self.g.num_attacks

    @property
    def num_assets(self) -> np.uintp:
        return self.g.num_services

    @property
    def num_defense_steps(self) -> np.uintp:
        return self.g.num_defenses

    def reset(self, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        if seed is None:
            seed = self.config.seed

        self.state = SimulatorState(self.g, self.config)
        return self.get_obs_dict(), self.info()

    def step(self, actions: Dict[str, int]) -> Tuple[Observation, Info]:
        funcs: Dict[str, Callable[[int], Tuple[NDArray[np.int8], bool]]] = {
            AGENT_ATTACKER: self.attack_step,
            AGENT_DEFENDER: self.enable_defense_step,
        }

        difference = np.zeros(self.g.num_attacks, dtype=np.int8)
        valid_actions = {AGENT_ATTACKER: False, AGENT_DEFENDER: False}

        for actor_id, action in actions.items():
            if action < len(self.special_actions):
                # Only special action is to wait for now
                valid_action = True
                agent_diff = np.zeros(self.g.num_attacks, dtype=np.int8)
            else:
                # offset by the number of special actions to get the step index
                step_index = action - len(self.special_actions)
                agent_diff, valid_action = funcs[actor_id](step_index)

            difference += agent_diff
            valid_actions[actor_id] = valid_action

        self.state.prev_actions = actions
        self.state.time += 1
        # Generate new noise so that FP and FN alerts change
        self.state.noise = self.state.rng.random(self.state.attack_state.shape, dtype=np.float64)
        self.state.valid_actions = valid_actions

        return self.get_obs_dict(), self.info()

    def get_obs_dict(self) -> Observation:
        return Observation(
            self.ids_observe(),
            self.state.attack_surface,
            self.state.defense_state,
            self.state.ttc_remaining,
            self.state.attack_state,
            defender_action_mask=self.get_defender_action_mask(),
            attacker_action_mask=self.get_attacker_action_mask(),
        )

    @property
    def ttc_total(self) -> UINT:
        return np.sum(self.state.ttc_remaining)

    def get_defender_action_mask(self) -> NDArray[np.int8]:
        action_mask = np.ones(self.g.num_defenses + self.num_special_actions, dtype=np.int8)
        action_mask[self.terminate_action] = 0
        action_mask[self.num_special_actions :] = self.state.defense_state
        return action_mask

    def get_attacker_action_mask(self) -> NDArray[np.int8]:
        action_mask = np.ones(self.g.num_attacks + self.num_special_actions, dtype=np.int8)
        action_mask[self.terminate_action] = 0
        action_mask[self.num_special_actions :] = self.state.attack_surface
        return action_mask

    def get_action_mask(self, agent: str) -> NDArray[np.int8]:
        funcs = {
            AGENT_ATTACKER: self.get_attacker_action_mask,
            AGENT_DEFENDER: self.get_defender_action_mask,
        }
        return funcs[agent]()

    def enable_defense_step(self, defense_idx: UINT) -> Tuple[NDArray[np.int8], bool]:
        """Enable (disable) a defense step."""

        # Only enable defenses that are disabled
        if not self.state.defense_state[defense_idx]:
            return np.zeros(self.g.num_attacks, dtype=np.int8), False

        # Enable (disable) the denfense step
        self.state.defense_state[defense_idx] = 0

        # Remove all affected attacks from the attack surface
        affected_steps = self.g.attack_steps_by_defense_step[defense_idx]
        self.state.attack_surface[affected_steps] = 0

        # Return the affected steps
        effect = np.zeros(self.g.num_attacks, dtype=np.int8)
        effect[affected_steps] = -1

        return effect, True

    def attack_step(self, attack_idx: UINT) -> Tuple[NDArray[np.int8], bool]:
        """Have the attacker perform an action."""

        # assert (
        #     attack_idx in self.valid_actions
        # ), "Attacker tried to perform an attack not in attack surface"

        if attack_idx not in np.flatnonzero(self.state.attack_surface):
            return np.zeros(self.g.num_attacks, dtype=np.int8), False

        # steps that the attacker compromised by performing this action
        compromised_steps: Set[UINT] = set()

        # If attack surface is empty, no need to perform an action
        if self.state.attack_surface_empty:
            return np.zeros(self.g.num_attacks, dtype=np.int8), True

        self.state.ttc_remaining[attack_idx] -= 1

        self.state.current_attack_step = self.g.attack_names[attack_idx]

        if self.state.ttc_remaining[attack_idx] != 0:
            return np.zeros(self.g.num_attacks, dtype=np.int8), True

        # successful attack, update reward, attack_state, attack_surface
        compromised_step = attack_idx
        compromised_steps.add(compromised_step)
        self.state.attack_state[compromised_step] = 1
        self.state.attack_surface[compromised_step] = 0

        # add reachable steps to the attack surface
        self.state.attack_surface[self.state.get_vulnerable_children(self.g, compromised_step)] = 1

        compromised_ass = self.compromise_steps()

        # recursively add reachable steps to the attack surface
        while len(compromised_ass) > 0:
            for step in compromised_ass:
                self.state.attack_surface[self.state.get_vulnerable_children(self.g, step)] = 1
                compromised_steps.add(step)
            compromised_ass = self.compromise_steps()

        effect = np.zeros(self.g.num_attacks, dtype=np.int8)
        effect[list(compromised_steps)] = 1

        return effect, True

    def compromise_steps(self) -> NDArray[np.int64]:
        """Set all steps with ttc=0 to compromised."""
        no_ttc = np.flatnonzero(self.state.ttc_remaining == 0)
        in_attack_surface = np.flatnonzero(self.state.attack_surface)
        compromised_ass = np.intersect1d(no_ttc, in_attack_surface)
        self.state.attack_state[compromised_ass] = 1
        self.state.attack_surface[compromised_ass] = 0
        return compromised_ass

    def info(self) -> Info:
        num_compromised_steps = len(self.state.compromised_steps)
        num_compromised_flags = len(self.compromised_flags)
        return Info(
            time=self.state.time,
            num_compromised_steps=num_compromised_steps,
            num_compromised_flags=num_compromised_flags,
            perc_compromised_flags=num_compromised_flags / self.num_flags,
            perc_compromised_steps=num_compromised_steps / self.num_attack_steps,
            perc_defenses_activated=np.sum(np.logical_not(self.state.defense_state))
            / self.num_defense_steps,
            num_observed_alerts=self.state.num_observed_alerts,
        )

    def interpret_defender_action(self, action: int) -> str:
        return (
            self.action_strings[action]
            if action in self.special_actions
            else self.g.defense_names[action]
        )

    def ids_observe(self) -> np.ndarray:
        """Observation of attack steps is subject to the true/false positive
        rates of an assumed underlying intrusion detection system Depending on
        the true and false positive rates for each step, ongoing attacks may
        not be reported, or non-existing attacks may be spuriously reported."""

        noisy_observation, self.state.false_positives, self.state.false_negatives = self.ids(
            self.state.attack_state, self.state.noise
        )

        return np.append(self.state.defense_state, noisy_observation)

    @property
    def compromised_flags(self) -> List[UINT]:
        return [flag for flag in self.g.flag_indices if self.state.attack_state[flag]]

    def dump_to_pickle(self, filename: str) -> None:
        if not filename:
            filename = f"sim_t={self.state.time}_dump.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)
