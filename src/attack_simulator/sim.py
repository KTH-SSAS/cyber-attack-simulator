from typing import List, Tuple

import numpy as np

from .config import EnvConfig, GraphConfig
from .graph import AttackGraph


ACTION_NAMES = ["wait", "use"]
class AttackSimulator:
    """Does the simulation."""

    NO_ACTION = 0
    ATTACKER_WAIT = -1
    NO_ACTION_STR = "nothing"

    def __init__(self, config: EnvConfig, rng: np.random.Generator) -> None:

        self.config = config
        self.rng = rng

        graph_config = config.graph_config if isinstance(config.graph_config, GraphConfig) else GraphConfig(**config.graph_config)

        self.g: AttackGraph = AttackGraph(graph_config)
        self.time = 0
        self.service_state = np.ones(self.g.num_services, dtype="int8")
        self.defense_state = np.ones(self.g.num_defenses, dtype="int8")
        self.attack_state = np.zeros(self.g.num_attacks, dtype="int8")
        self.attack_surface = np.zeros(self.g.num_attacks, dtype="int8")
        self.false_negative = config.false_negative
        self.false_positive = config.false_positive

        # Initial state
        self.entry_attack_index = self.g.attack_indices[self.g.root]

        self.ttc_remaining = np.array(
            [max(1, int(v)) for v in self.rng.exponential(self.g.ttc_params)]
        )
        self.ttc_total = sum(self.ttc_remaining)

        # Set the TTC for the entry attack to be the attack start time
        self.attack_start_time = int(self.rng.exponential(self.config.attack_start_time))
        self.ttc_remaining[self.entry_attack_index] = self.attack_start_time

        if self.attack_start_time == 0:
            self.attack_surface[self.entry_attack_index] = 0
            self.attack_state[self.entry_attack_index] = 1
            # add reachable steps to the attack surface
            self.attack_surface[self._get_reachable_steps(self.entry_attack_index)] = 1
        else:
            self.attack_surface[self.entry_attack_index] = 1

        self.attacker_action: int = self.entry_attack_index
        self.defender_action: int = -1
        self.defender_target: int = -1
        self.last_observation = None

        self.noise = self.generate_noise()

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

    @property
    def attack_started(self) -> bool:
        return self.time >= self.attack_start_time

    def defense_action(self, action_type: int, target: int) -> bool:
        """Enable (disable) a defense step."""

        self.defender_action = action_type
        self.defender_target = target

        if action_type == self.NO_ACTION:
            return False

        # Only enable defenses that are disabled
        if not self.defense_state[target]:
            return False

        # Enable (disable) the denfense step
        self.defense_state[target] = 0

        # Remove all affected attacks from the attack surface
        affected_steps = self.g.attack_steps_by_defense_step[target]
        self.attack_surface[affected_steps] = 0

        return False

    @property
    def valid_actions(self) -> np.ndarray:
        return np.flatnonzero(self.attack_surface)

    def attack_action(self, attacker_action: int) -> bool:
        """Have the attacker perform an action."""

        # If attack surface is empty, no need to perform an action
        # Currently, if the attacker waits then the attack surface is empty.
        self.attacker_action = attacker_action
        if self.attack_surface_empty or attacker_action == self.ATTACKER_WAIT:
            return True

        assert (
            attacker_action in self.valid_actions
        ), "Attacker tried to perform an attack not in attack surface"

        self.ttc_remaining[attacker_action] -= 1

        if self.ttc_remaining[attacker_action] != 0:
            return False

        # successful attack, update reward, attack_state, attack_surface
        self.attack_state[attacker_action] = 1
        self.attack_surface[attacker_action] = 0

        # add reachable steps to the attack surface
        self.attack_surface[self._get_reachable_steps(attacker_action)] = 1

        # end episode when attack surface becomes empty
        done = self.attack_surface_empty
        return done

    def _get_reachable_steps(self, attack_index: int) -> List[int]:
        return self.g.get_reachable_steps(attack_index, self.attack_state, self.defense_state)

    def step(self) -> bool:
        self.time += 1

        # Generate new noise so that FP and FN alerts change
        self.noise = self.generate_noise()

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

    def interpret_defender_target(self, target: int) -> str:
        return self.g.defense_names[target]

    def interpret_defender_action(self, action: int) -> str:
        return ACTION_NAMES[action]

    def generate_noise(self) -> np.ndarray:
        """Generates a "noise" mask to use for false positives and
        negatives."""
        return self.rng.uniform(0, 1, self.num_attack_steps)

    def observe(self) -> np.ndarray:
        """Observation of attack steps is subject to the true/false positive
        rates of an assumed underlying intrusion detection system Depending on
        the true and false positive rates for each step, ongoing attacks may
        not be reported, or non-existing attacks may be spuriously reported."""
        probabilities = self.noise
        false_negatives = self.attack_state & (probabilities >= self.false_negative)
        false_positives = (1 - self.attack_state) & (probabilities <= self.false_positive)
        detected = false_negatives | false_positives
        return np.append(self.defense_state, detected)

    def current_attack_step(self) -> Tuple[str, int]:
        """Returns name of the attack step the attacker is currently
        targeting."""
        current_step = (
            self.NO_ACTION_STR
            if self.attacker_action == self.ATTACKER_WAIT
            else self.g.attack_names[self.attacker_action]
        )
        ttc_remaining = (
            0
            if self.attacker_action == self.ATTACKER_WAIT
            else self.ttc_remaining[self.attacker_action]
        )
        return current_step, ttc_remaining

    @property
    def compromised_steps(self) -> np.ndarray:
        return np.flatnonzero(self.attack_state)

    @property
    def compromised_flags(self) -> List[int]:
        return [flag for flag in self.g.flags if self.attack_state[flag]]
