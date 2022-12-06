from typing import Callable, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import EnvConfig, GraphConfig
from .graph import AttackGraph

from .ids import ProbabilityIDS
import pickle

class AttackSimulator:
    """Does the simulation."""

    NO_ACTION = -1
    NO_ACTION_STR = "nothing"

    def __init__(self, config: EnvConfig, seed: int) -> None:

        self.config = config
        self.rng = np.random.default_rng(seed)

        self.ids = ProbabilityIDS(seed, config.false_negative, config.false_positive)

        graph_config = (
            config.graph_config
            if isinstance(config.graph_config, GraphConfig)
            else GraphConfig(**config.graph_config)
        )

        self.g: AttackGraph = AttackGraph(graph_config)
        self.time = 0
        self.service_state = np.ones(self.g.num_services, dtype="int8")
        self.defense_state = np.ones(self.g.num_defenses, dtype="int8")
        self.attack_state = np.zeros(self.g.num_attacks, dtype="int8")
        self.attack_surface = np.zeros(self.g.num_attacks, dtype="int8")
        self.fnr = config.false_negative
        self.fpr = config.false_positive

        self.false_positives = np.zeros(self.g.num_attacks, dtype="int8")
        self.false_negatives = np.zeros(self.g.num_attacks, dtype="int8")

        self.last_observation = np.zeros(self.g.num_attacks, dtype="int8")

        self.num_observed_alerts = 0
        self.num_alerts = 0

        # Initial state
        self.entry_attack_index = self.g.attack_indices[self.g.root]

        if config.randomize_ttc:
            self.ttc_remaining = np.array(
                [v if v == 0 else max(1, int(v)) for v in self.rng.exponential(self.g.ttc_params)]
            )
        else:
            self.ttc_remaining = self.g.ttc_params

        self.ttc_total = sum(self.ttc_remaining)

        # Set the TTC for the entry attack to be the attack start time
        self.attack_start_time = int(self.rng.exponential(self.config.attack_start_time))
        self.ttc_remaining[self.entry_attack_index] = self.attack_start_time

        if self.attack_start_time == 0:
            self.attack_surface[self.entry_attack_index] = 0
            self.attack_state[self.entry_attack_index] = 1
            # add reachable steps to the attack surface
            self.attack_surface[self._get_vulnerable_children(self.entry_attack_index)] = 1
        else:
            self.attack_surface[self.entry_attack_index] = 1

        self.attacker_action: int = self.entry_attack_index
        self.defender_action: int = self.NO_ACTION

        self.noise = self.rng.random(self.attack_state.shape)

    @property
    def num_flags(self) -> int:
        return len(self.g.flags)

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

    def defense_action(self, defender_action: int) -> bool:
        """Enable (disable) a defense step."""

        self.defender_action = defender_action

        if defender_action == self.NO_ACTION:
            return False

        # Only enable defenses that are disabled
        if not self.defense_state[defender_action]:
            return False

        # Enable (disable) the denfense step
        self.defense_state[defender_action] = 0

        # Remove all affected attacks from the attack surface
        affected_steps = self.g.attack_steps_by_defense_step[defender_action]
        self.attack_surface[affected_steps] = 0

        return False

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

    def step(self) -> bool:
        self.time += 1

        # Generate new noise so that FP and FN alerts change
        self.noise = self.rng.random(self.attack_state.shape)

        # Log alerts
        self.num_observed_alerts += self.last_observation.sum()
        self.num_alerts += self.attack_state.sum()

        # Nothing here to end the episode yet

        return False

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
        return [flag for flag in self.g.flags if self.attack_state[flag]]


    def dump_to_pickle(self, filename: str) -> None:
        if not filename:
            filename = f"sim_t={self.time}_dump.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f) 
