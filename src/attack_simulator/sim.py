from typing import Callable, List, Set, Tuple, Dict, Any

import numpy as np
from numpy.typing import NDArray

from .config import EnvConfig
from .graph import AttackGraph

from .ids import ProbabilityIDS
import pickle


class AttackSimulator:
    """Does the simulation."""

    NO_ACTION = -1
    NO_ACTION_STR = "nothing"

    ATTACKER_INDEX = 0
    DEFENDER_INDEX = 1

    def __init__(self, config: EnvConfig, attack_graph: AttackGraph, seed: int) -> None:

        ## Constants
        self.config = config

        self.ids = ProbabilityIDS(seed, config.false_negative, config.false_positive)
        self.fnr = config.false_negative
        self.fpr = config.false_positive
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
            self.num_alerts,
            self.attacker_action,
            self.defender_action,
        ) = self.get_initial_state(seed)

        # Set the TTC for the entry attack to be the attack start time
        self.ttc_remaining, self.ttc_total = self.get_initial_ttc_vals(self.attack_start_time)
        self.attack_surface = self.get_initial_attack_surface(self.attack_start_time)

    def get_initial_attack_surface(self, attack_start_time: int) -> NDArray:
        attack_surface = np.zeros(self.g.num_attacks, dtype="int8")
        if attack_start_time == 0:
            attack_surface[self.entry_attack_index] = 0
            self.attack_state[self.entry_attack_index] = 1
            # add reachable steps to the attack surface
            attack_surface[self._get_vulnerable_children(self.entry_attack_index)] = 1
        else:
            attack_surface[self.entry_attack_index] = 1

        return attack_surface

    def get_initial_ttc_vals(self, attack_start_time: int) -> Tuple[NDArray[np.int64], int]:
        if self.config.randomize_ttc:
            ttc_remaining = np.array(
                [v if v == 0 else max(1, int(v)) for v in self.rng.exponential(self.g.ttc_params)],
                dtype=np.int64,
            )
        else:
            ttc_remaining = np.array(self.g.ttc_params, dtype=np.int64)

        ttc_remaining[self.entry_attack_index] = attack_start_time

        return ttc_remaining, sum(ttc_remaining)

    @property
    def num_flags(self) -> int:
        return len(self.g.flag_indices)

    @property
    def attacker_observation(self) -> NDArray:
        return np.stack([self.attack_surface, self.attack_state])

    @property
    def num_attack_steps(self) -> int:
        return self.g.num_attacks

    @property
    def num_assets(self) -> int:
        return self.g.num_services

    @property
    def num_defense_steps(self) -> int:
        return self.g.num_defenses

    @property
    def attack_surface_empty(self) -> bool:
        return not any(self.attack_surface)

    def reset(self, seed: int) -> Tuple[NDArray, Dict[str, Any]]:
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
            self.num_alerts,
            self.attacker_action,
            self.defender_action,
        ) = self.get_initial_state(seed)

        self.ttc_remaining, self.ttc_total = self.get_initial_ttc_vals(self.attack_start_time)
        self.attack_surface = self.get_initial_attack_surface(self.attack_start_time)
        return self.observe(), self.info()


    def get_initial_state(
        self, seed: int
    ) -> Tuple[
        int,
        NDArray[np.int8],
        NDArray[np.int8],
        NDArray[np.int8],
        NDArray[np.int8],
        NDArray[np.int8],
        NDArray[np.int8],
        np.random.Generator,
        NDArray[np.float64],
        int,
        int,
        int,
        int,
        int,
    ]:
        return (
            0,
            np.ones(self.g.num_services, dtype="int8"),
            np.ones(self.g.num_defenses, dtype="int8"),
            np.zeros(self.g.num_attacks, dtype="int8"),
            np.zeros(self.g.num_attacks, dtype="int8"),
            np.zeros(self.g.num_attacks, dtype="int8"),
            np.zeros(self.g.num_attacks, dtype="int8"),
            np.random.default_rng(seed),
            np.random.default_rng(seed).random(self.g.num_attacks, dtype=np.float64),
            int(np.random.default_rng(seed).exponential(self.config.attack_start_time)),
            0,
            0,
            self.entry_attack_index,
            self.NO_ACTION,
        )

    def defense_action(self, defender_action: int) -> Tuple[bool, NDArray[np.int64]]:
        """Enable (disable) a defense step."""

        self.defender_action = defender_action

        if defender_action == self.NO_ACTION:
            return False, self.observe()

        # Only enable defenses that are disabled
        if not self.defense_state[defender_action]:
            return False, self.observe()

        # Enable (disable) the denfense step
        self.defense_state[defender_action] = 0

        # Remove all affected attacks from the attack surface
        affected_steps = self.g.attack_steps_by_defense_step[defender_action]
        self.attack_surface[affected_steps] = 0

        return False, self.observe()

    @property
    def valid_actions(self) -> np.ndarray:
        return np.flatnonzero(self.attack_surface)

    def attack_action(self, attacker_action: int) -> Tuple[bool, NDArray[np.int64]]:
        """Have the attacker perform an action."""

        # steps that the attacker compromised by performing this action

        set2array: Callable[[Set[int]], NDArray[np.int64]] = lambda x: np.array(
            list(x), dtype=np.int64
        )

        compromised_steps: Set[int] = set()

        # If attack surface is empty, no need to perform an action
        if self.attack_surface_empty:
            return True, set2array(compromised_steps)

        self.attacker_action = attacker_action

        if attacker_action == self.NO_ACTION:
            return False, set2array(compromised_steps)

        assert (
            attacker_action in self.valid_actions
        ), "Attacker tried to perform an attack not in attack surface"

        self.ttc_remaining[attacker_action] -= 1

        if self.ttc_remaining[attacker_action] != 0:
            return False, set2array(compromised_steps)

        # successful attack, update reward, attack_state, attack_surface
        compromised_step = attacker_action
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

        # end episode when attack surface becomes empty
        done = self.attack_surface_empty
        return done, set2array(compromised_steps)

    def compromise_steps(self) -> NDArray[np.int64]:
        """Set all steps with ttc=0 to compromised."""
        no_ttc = np.flatnonzero(self.ttc_remaining == 0)
        in_attack_surface = np.flatnonzero(self.attack_surface)
        compromised_ass = np.intersect1d(no_ttc, in_attack_surface)
        self.attack_state[compromised_ass] = 1
        self.attack_surface[compromised_ass] = 0
        return compromised_ass

    def _get_vulnerable_children(self, attack_index: int) -> List[int]:
        return self.g.get_vulnerable_children(attack_index, self.attack_state, self.defense_state)

    def step(self) -> Tuple[bool, Dict[str, Any]]:
        self.time += 1

        # Generate new noise so that FP and FN alerts change
        self.noise = self.rng.random(self.attack_state.shape, dtype=np.float64)

        # Log alerts
        self.num_observed_alerts += self.last_observation.sum()
        self.num_alerts += self.attack_state.sum()

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
            "num_alerts": self.num_alerts,
        }

    def interpret_services(self, services: np.ndarray) -> List[str]:
        return list(np.array(self.g.service_names)[np.flatnonzero(services)])

    def interpret_defenses(self, active_defenses: np.ndarray) -> List[str]:
        return [name for name, state in zip(self.g.defense_names, active_defenses) if not state]

    def interpret_attacks(self, attacks: np.ndarray) -> List[str]:
        return list(np.array(self.g.attack_names)[np.flatnonzero(attacks)])

    def interpret_observation(self, observation: np.ndarray) -> Tuple[List[str], List[str]]:
        defenses = observation[: self.g.num_defenses]
        attacks = observation[self.g.num_defenses :]
        return self.interpret_defenses(defenses), self.interpret_attacks(attacks)

    def interpret_defender_action(self, action: int) -> str:
        return self.NO_ACTION_STR if action == self.NO_ACTION else self.g.defense_names[action]

    def observe(self) -> np.ndarray:
        """Observation of attack steps is subject to the true/false positive
        rates of an assumed underlying intrusion detection system Depending on
        the true and false positive rates for each step, ongoing attacks may
        not be reported, or non-existing attacks may be spuriously reported."""

        noisy_observation, self.false_positives, self.false_negatives = self.ids(
            self.attack_state, self.noise
        )

        return np.append(self.defense_state, noisy_observation)

    def current_attack_step(self) -> Tuple[str, int]:
        """Returns name of the attack step the attacker is currently
        targeting."""
        current_step = (
            self.NO_ACTION_STR
            if self.attacker_action == self.NO_ACTION
            else self.g.attack_names[self.attacker_action]
        )
        ttc_remaining = (
            0
            if self.attacker_action == self.NO_ACTION
            else self.ttc_remaining[self.attacker_action]
        )
        return current_step, ttc_remaining

    @property
    def compromised_steps(self) -> np.ndarray:
        return np.flatnonzero(self.attack_state)

    @property
    def compromised_flags(self) -> List[int]:
        return [flag for flag in self.g.flag_indices if self.attack_state[flag]]

    def dump_to_pickle(self, filename: str) -> None:
        if not filename:
            filename = f"sim_t={self.time}_dump.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)
