import pickle
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from attack_simulator.constants import (
    ACTION_STRINGS,
    AGENT_ATTACKER,
    AGENT_DEFENDER,
    UINT,
    special_actions,
)
from attack_simulator.observation import Info, Observation
from attack_simulator.rng import get_rng

from .config import SimulatorConfig
from .graph import AttackGraph
from .ids import ProbabilityIDS


class AttackSimulator:
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

        ## State
        (
            self.time,
            self.service_state,
            self.defense_state,
            self.attack_state,
            self.false_positives,
            self.false_negatives,
            self.last_observation,
            self.rng,
            self.noise,
            self.attack_start_time,
            self.num_observed_alerts,
            self.prev_actions,
            self.valid_actions,
        ) = self.get_initial_state(config.seed)

        # Set the TTC for the entry attack to be the attack start time
        self.ttc_remaining, self.ttc_total = self.get_initial_ttc_vals(self.attack_start_time)
        self.attack_surface = self.get_initial_attack_surface(self.attack_start_time)

    def get_initial_attack_surface(self, attack_start_time: UINT) -> NDArray:
        attack_surface = np.zeros(self.g.num_attacks, dtype="int8")
        if attack_start_time == 0:
            attack_surface[self.entry_attack_index] = 0
            self.attack_state[self.entry_attack_index] = 1
            # add reachable steps to the attack surface
            attack_surface[self._get_vulnerable_children(self.entry_attack_index)] = 1
        else:
            attack_surface[self.entry_attack_index] = 1

        return attack_surface

    def get_initial_ttc_vals(self, attack_start_time: UINT) -> Tuple[NDArray[UINT], UINT]:
        if self.config.randomize_ttc:
            ttc_remaining = np.array(
                [v if v == 0 else max(1, UINT(v)) for v in self.rng.exponential(self.g.ttc_params)],
                dtype=UINT,
            )
        else:
            ttc_remaining = np.array(self.g.ttc_params, dtype=UINT)

        ttc_remaining[self.entry_attack_index] = attack_start_time

        return ttc_remaining, np.sum(ttc_remaining)

    @property
    def num_flags(self) -> int:
        return self.g.flag_indices.shape[0]

    @property
    def attacker_observation(self) -> NDArray:
        return np.stack([self.attack_surface, self.attack_state])

    @property
    def num_attack_steps(self) -> UINT:
        return self.g.num_attacks

    @property
    def num_assets(self) -> np.uintp:
        return self.g.num_services

    @property
    def num_defense_steps(self) -> np.uintp:
        return self.g.num_defenses

    @property
    def attack_surface_empty(self) -> bool:
        return not any(self.attack_surface)

    @property
    def valid_attacks(self) -> NDArray[np.intp]:
        return np.flatnonzero(self.attack_surface)

    @property
    def valid_defenses(self) -> NDArray[np.intp]:
        return np.flatnonzero(self.defense_state)

    def reset(self, seed: Optional[UINT] = None) -> Tuple[Observation, Info]:
        if seed is None:
            seed = self.config.seed

        (
            self.time,
            self.service_state,
            self.defense_state,
            self.attack_state,
            self.false_positives,
            self.false_negatives,
            self.last_observation,
            self.rng,
            self.noise,
            self.attack_start_time,
            self.num_observed_alerts,
            self.prev_actions,
            self.valid_actions,
        ) = self.get_initial_state(seed)

        self.ttc_remaining, self.ttc_total = self.get_initial_ttc_vals(self.attack_start_time)
        self.attack_surface = self.get_initial_attack_surface(self.attack_start_time)
        return self.get_obs_dict(), self.info()

    def get_initial_state(
        self, seed: int
    ) -> Tuple[
        UINT,
        NDArray[np.int8],
        NDArray[np.int8],
        NDArray[np.int8],
        NDArray[np.int8],
        NDArray[np.int8],
        NDArray[np.int8],
        np.random.Generator,
        NDArray[np.float64],
        UINT,
        UINT,
        OrderedDict[str, UINT],
        Dict[str, bool],
    ]:
        rng = get_rng(seed)[0]
        return (
            0,  # time
            np.ones(self.g.num_services, dtype="int8"),  # service state
            np.ones(self.g.num_defenses, dtype="int8"),  # defense state
            np.zeros(self.g.num_attacks, dtype="int8"),  # attack state
            np.zeros(self.g.num_attacks, dtype="int8"),  # false positives
            np.zeros(self.g.num_attacks, dtype="int8"),  # false negatives
            np.zeros(self.g.num_attacks, dtype="int8"),  # last observation
            rng,  # rng
            rng.random(self.g.num_attacks, dtype=np.float64),  # noise
            UINT(rng.exponential(self.config.attack_start_time)),  # attack start time
            0,  # num observed alerts
            OrderedDict([(AGENT_ATTACKER, self.entry_attack_index), (AGENT_DEFENDER, UINT(0))]),
            {AGENT_ATTACKER: False, AGENT_DEFENDER: False},
        )

    def step(self, actions: OrderedDict[str, UINT]) -> Tuple[Observation, Info]:
        self.prev_actions = actions

        funcs: Dict[str, Callable[[UINT], Tuple[NDArray[np.int8], bool]]] = {
            AGENT_ATTACKER: self.attack_step,
            AGENT_DEFENDER: self.enable_defense_step,
        }

        difference = np.zeros(self.g.num_attacks, dtype=np.int8)
        valid_actions = {AGENT_ATTACKER: False, AGENT_DEFENDER: False}

        for actor_id, action in actions.items():

            if action < len(special_actions):
                # Only special action is to wait for now
                valid_action = True
                agent_diff = np.zeros(self.g.num_attacks, dtype=np.int8)
            else:
                step_index = action - len(special_actions)
                agent_diff, valid_action = funcs[actor_id](step_index)

            difference += agent_diff
            valid_actions[actor_id] = valid_action

        self.time += 1
        # Generate new noise so that FP and FN alerts change
        self.noise = self.rng.random(self.attack_state.shape, dtype=np.float64)
        self.valid_actions = valid_actions

        return self.get_obs_dict(), self.info()

    def get_obs_dict(self) -> Observation:
        return Observation(
            self.ids_observe(),
            self.attack_surface,
            self.defense_state,
            self.ttc_remaining,
            self.attack_state,
        )

    def enable_defense_step(self, defense_idx: UINT) -> Tuple[NDArray[np.int8], bool]:
        """Enable (disable) a defense step."""

        # Only enable defenses that are disabled
        if not self.defense_state[defense_idx]:
            return np.zeros(self.g.num_attacks, dtype=np.int8), False

        # Enable (disable) the denfense step
        self.defense_state[defense_idx] = 0

        # Remove all affected attacks from the attack surface
        affected_steps = self.g.attack_steps_by_defense_step[defense_idx]
        self.attack_surface[affected_steps] = 0

        # Return the affected steps
        effect = np.zeros(self.g.num_attacks, dtype=np.int8)
        effect[affected_steps] = -1

        return effect, True

    def attack_step(self, attack_idx: UINT) -> Tuple[NDArray[np.int8], bool]:
        """Have the attacker perform an action."""

        # assert (
        #     attack_idx in self.valid_actions
        # ), "Attacker tried to perform an attack not in attack surface"

        if attack_idx not in np.flatnonzero(self.attack_surface):
            return np.zeros(self.g.num_attacks, dtype=np.int8), False

        # steps that the attacker compromised by performing this action
        compromised_steps: Set[UINT] = set()

        # If attack surface is empty, no need to perform an action
        if self.attack_surface_empty:
            return np.zeros(self.g.num_attacks, dtype=np.int8), True

        self.ttc_remaining[attack_idx] -= 1

        if self.ttc_remaining[attack_idx] != 0:
            return np.zeros(self.g.num_attacks, dtype=np.int8), True

        # successful attack, update reward, attack_state, attack_surface
        compromised_step = attack_idx
        compromised_steps.add(compromised_step)
        self.attack_state[compromised_step] = 1
        self.attack_surface[compromised_step] = 0

        # add reachable steps to the attack surface
        self.attack_surface[self._get_vulnerable_children(compromised_step)] = 1

        compromised_ass = self.compromise_steps()

        # recursively add reachable steps to the attack surface
        while len(compromised_ass) > 0:
            for step in compromised_ass:
                self.attack_surface[self._get_vulnerable_children(step)] = 1
                compromised_steps.add(step)
            compromised_ass = self.compromise_steps()

        effect = np.zeros(self.g.num_attacks, dtype=np.int8)
        effect[list(compromised_steps)] = 1

        return effect, True

    def compromise_steps(self) -> NDArray[UINT]:
        """Set all steps with ttc=0 to compromised."""
        no_ttc = np.flatnonzero(self.ttc_remaining == 0)
        in_attack_surface = np.flatnonzero(self.attack_surface)
        compromised_ass = np.intersect1d(no_ttc, in_attack_surface)
        self.attack_state[compromised_ass] = 1
        self.attack_surface[compromised_ass] = 0
        return compromised_ass

    def _get_vulnerable_children(self, attack_index: UINT) -> List[UINT]:
        return self.g.get_vulnerable_children(attack_index, self.attack_state, self.defense_state)

    def info(self) -> Info:
        return Info(
            self.time,
            len(self.compromised_steps),
            len(self.compromised_flags),
            len(self.compromised_flags) / self.num_flags,
            len(self.compromised_steps) / self.num_attack_steps,
            sum(np.logical_not(self.defense_state)) / self.num_defense_steps,
            self.num_observed_alerts,
        )

    def interpret_defender_action(self, action: int) -> str:
        return ACTION_STRINGS[action] if action in special_actions else self.g.defense_names[action]

    def ids_observe(self) -> np.ndarray:
        """Observation of attack steps is subject to the true/false positive
        rates of an assumed underlying intrusion detection system Depending on
        the true and false positive rates for each step, ongoing attacks may
        not be reported, or non-existing attacks may be spuriously reported."""

        noisy_observation, self.false_positives, self.false_negatives = self.ids(
            self.attack_state, self.noise
        )

        return np.append(self.defense_state, noisy_observation)

    def current_attack_step(self) -> Tuple[str, UINT]:
        """Returns name of the attack step the attacker is currently
        targeting."""
        action = self.prev_actions[AGENT_ATTACKER]
        is_special_action = action in special_actions
        current_step = (
            ACTION_STRINGS[action]
            if is_special_action
            else self.g.attack_names[action - len(special_actions)]
        )
        ttc_remaining = (
            0 if is_special_action else self.ttc_remaining[action - len(special_actions)]
        )
        return current_step, ttc_remaining

    @property
    def compromised_steps(self) -> np.ndarray:
        return np.flatnonzero(self.attack_state)

    @property
    def compromised_flags(self) -> List[UINT]:
        return [flag for flag in self.g.flag_indices if self.attack_state[flag]]

    def dump_to_pickle(self, filename: str) -> None:
        if not filename:
            filename = f"sim_t={self.time}_dump.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)
