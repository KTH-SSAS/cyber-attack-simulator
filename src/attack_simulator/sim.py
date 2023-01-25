import pickle
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from attack_simulator.constants import (
    ACTION_STRINGS,
    AGENT_ATTACKER,
    AGENT_DEFENDER,
    UINT,
    special_actions,
)

from .config import SimulatorConfig
from .graph import AttackGraph
from .ids import ProbabilityIDS


class AttackSimulator:
    """Does the simulation."""

    def __init__(self, config: SimulatorConfig, attack_graph: AttackGraph, seed: UINT) -> None:
        ## Constants
        self.config = config

        self.ids = ProbabilityIDS(seed, config.false_negative_rate, config.false_positive_rate)
        self.fnr = config.false_negative_rate
        self.fpr = config.false_positive_rate
        self.g: AttackGraph = attack_graph
        self.entry_attack_index = self.g.attack_indices[self.g.root]

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
        ) = self.get_initial_state(seed)

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

        return ttc_remaining, sum(ttc_remaining)

    @property
    def num_flags(self) -> UINT:
        return len(self.g.flag_indices)

    @property
    def attacker_observation(self) -> NDArray:
        return np.stack([self.attack_surface, self.attack_state])

    @property
    def num_attack_steps(self) -> UINT:
        return self.g.num_attacks

    @property
    def num_assets(self) -> UINT:
        return self.g.num_services

    @property
    def num_defense_steps(self) -> UINT:
        return self.g.num_defenses

    @property
    def attack_surface_empty(self) -> bool:
        return not any(self.attack_surface)

    def reset(self, seed: UINT) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        ) = self.get_initial_state(seed)

        self.ttc_remaining, self.ttc_total = self.get_initial_ttc_vals(self.attack_start_time)
        self.attack_surface = self.get_initial_attack_surface(self.attack_start_time)
        return self.get_obs_dict(None), self.info()

    def get_initial_state(
        self, seed: UINT
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
        Dict[str, UINT],
    ]:
        return (
            0,  # time
            np.ones(self.g.num_services, dtype="int8"),  # service state
            np.ones(self.g.num_defenses, dtype="int8"),  # defense state
            np.zeros(self.g.num_attacks, dtype="int8"),  # attack state
            np.zeros(self.g.num_attacks, dtype="int8"),  # false positives
            np.zeros(self.g.num_attacks, dtype="int8"),  # false negatives
            np.zeros(self.g.num_attacks, dtype="int8"),  # last observation
            np.random.default_rng(seed),  # rng
            np.random.default_rng(seed).random(self.g.num_attacks, dtype=np.float64),  # noise
            UINT(
                np.random.default_rng(seed).exponential(self.config.attack_start_time)
            ),  # attack start time
            0,  # num observed alerts
            {AGENT_ATTACKER: self.entry_attack_index, AGENT_DEFENDER: 0},
        )

    def action(self, action: UINT, actor_id: str) -> Dict[str, Any]:
        self.prev_actions[actor_id] = action

        if action < len(special_actions):
            return self.get_obs_dict(None)

        step_index = action - len(special_actions)

        if actor_id == AGENT_ATTACKER:
            affected_steps = self.attack_step(step_index)
        elif actor_id == AGENT_DEFENDER:
            affected_steps = self.enable_defense_step(step_index)
        else:
            raise ValueError("Invalid actor id")

        return self.get_obs_dict(affected_steps)

    def get_obs_dict(self, affected_steps: Optional[UINT] = None) -> Dict[str, Any]:
        return {
            "defender_obs": self.observe(),
            "attack_surface": self.attack_surface,
            "affected_steps": [] if affected_steps is None else affected_steps,
        }

    def enable_defense_step(self, defense_idx: UINT) -> NDArray[UINT]:
        """Enable (disable) a defense step."""

        # Only enable defenses that are disabled
        if not self.defense_state[defense_idx]:
            return np.array([], dtype=UINT)

        # Enable (disable) the denfense step
        self.defense_state[defense_idx] = 0

        # Remove all affected attacks from the attack surface
        affected_steps = self.g.attack_steps_by_defense_step[defense_idx]
        self.attack_surface[affected_steps] = 0

        return np.array(affected_steps, dtype=UINT)

    @property
    def valid_actions(self) -> np.ndarray:
        return np.flatnonzero(self.attack_surface)

    def attack_step(self, attack_idx: UINT) -> NDArray[np.uint64]:
        """Have the attacker perform an action."""

        # assert (
        #     attack_idx in self.valid_actions
        # ), "Attacker tried to perform an attack not in attack surface"

        if attack_idx not in self.valid_actions:
            return np.array([], dtype=UINT)

        # steps that the attacker compromised by performing this action

        set2array: Callable[[Set[UINT]], NDArray[np.uint64]] = lambda x: np.array(
            list(x), dtype=UINT
        )

        compromised_steps: Set[UINT] = set()

        # If attack surface is empty, no need to perform an action
        if self.attack_surface_empty:
            return set2array(compromised_steps)

        self.ttc_remaining[attack_idx] -= 1

        if self.ttc_remaining[attack_idx] != 0:
            return set2array(compromised_steps)

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

        return set2array(compromised_steps)

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

    def step(self) -> Tuple[bool, Dict[str, Any]]:
        self.time += 1

        # Generate new noise so that FP and FN alerts change
        self.noise = self.rng.random(self.attack_state.shape, dtype=np.float64)

        # Log alerts
        self.num_observed_alerts += self.last_observation.sum()
        # self.num_alerts += self.attack_state.sum()

        info = self.info()

        # Nothing here to end the episode yet
        return False, info

    def info(self) -> Dict[str, Any]:
        current_step, ttc_remaining = self.current_attack_step()
        return {
            "time": self.time,
            "current_step": current_step,
            "ttc_remaining_on_current_step": ttc_remaining,
            "attack_surface": self.attack_surface,
            "attack_state": self.attack_state,
            "defense_state": self.defense_state,
            "num_compromised_steps": len(self.compromised_steps),
            "num_compromised_flags": len(self.compromised_flags),
            "perc_compromised_steps": len(self.compromised_steps) / self.num_attack_steps,
            "perc_compromised_flags": len(self.compromised_flags) / self.num_flags,
            "perc_defenses_activated": sum(np.logical_not(self.defense_state))
            / self.num_defense_steps,
        }

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the episode."""
        return {
            "attacker_start_time": self.attack_start_time,
            "num_defenses": self.num_defense_steps,
            "num_attack_steps": self.num_attack_steps,
            "defense_costs": self.g.defense_costs,
            "flag_costs": self.g.flag_rewards,
            "num_observed_alerts": self.num_observed_alerts,
        }

    def interpret_defender_action(self, action: UINT) -> str:
        return ACTION_STRINGS[action] if action in special_actions else self.g.defense_names[action]

    def observe(self) -> np.ndarray:
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
