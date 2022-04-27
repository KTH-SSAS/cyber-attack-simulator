import numpy as np

from .config import EnvConfig
from .graph import AttackGraph


class AttackSimulator:
    """Does the simulation."""

    NO_ACTION = 0
    NO_ACTION_STR = "no action"

    def __init__(self, config: EnvConfig, rng) -> None:

        self.config = config
        self.rng = rng
        self.g: AttackGraph = AttackGraph(config.graph_config)
        self.time = 0
        self.service_state = np.ones(self.g.num_services, dtype="int8")
        self.defense_state = np.ones(self.g.num_defenses, dtype="int8")
        self.attack_state = np.zeros(self.g.num_attacks, dtype="int8")
        self.attack_surface = np.zeros(self.g.num_attacks, dtype="int8")
        self.false_negative = config.false_negative
        self.false_positive = config.false_positive

        # Initial state
        self.entry_attack_index = self.g.attack_indices[self.g.root]
        self.attack_surface[self.entry_attack_index] = 1

        self.ttc_remaining = np.array(
            [max(1, int(v)) for v in self.rng.exponential(self.g.ttc_params)]
        )
        self.ttc_total = sum(self.ttc_remaining)

        self.attack_index = self.entry_attack_index
        self.defender_action = self.NO_ACTION
        self.done = False
        self.last_observation = None

        self.noise = self.generate_noise()

    @property
    def num_attack_steps(self):
        return self.g.num_attacks

    @property
    def num_assets(self):
        return self.g.num_services

    @property
    def num_defense_steps(self):
        return self.g.num_defenses

    def defense_action(self, defense_index):
        """Enable (disable) a defense step."""
        self.defender_action = defense_index

        done = False
        # Only enable defenses that are disabled
        if self.defense_state[defense_index]:
            # Enable (disable) the denfense step
            self.defense_state[defense_index] = 0

            # Remove all affected attacks from the attack surface
            affected_steps = self.g.attack_steps_by_defense_step[defense_index]
            self.attack_surface[affected_steps] = 0

            # end episode when attack surface becomes empty
            done = not any(self.attack_surface)

        self.done = done
        return done

    @property
    def valid_actions(self):
        return np.flatnonzero(self.attack_surface)

    def attack_action(self, action):
        """Have the attacker perform an action."""
        done = False

        assert (
            action in self.valid_actions
        ), "Attacker tried to perform an attack not in attack surface"
        self.attack_index = action

        self.ttc_remaining[action] -= 1
        if self.ttc_remaining[action] == 0:
            # successful attack, update reward, attack_state, attack_surface
            self.attack_state[action] = 1
            self.attack_surface[action] = 0

            # add reachable steps to the attack surface
            self.attack_surface[self._get_reachable_steps(action)] = 1

            # end episode when attack surface becomes empty
            done = not any(self.attack_surface)

        self.done = done
        return done

    def _get_reachable_steps(self, attack_index):
        return self.g.get_reachable_steps(attack_index, self.attack_state, self.defense_state)

    def step(self):
        self.time += 1

        # Generate new noise so that FP and FN alerts change
        self.noise = self.generate_noise()

    def interpret_services(self, services=None):
        if services is None:
            services = self.service_state
        return list(np.array(self.g.service_names)[np.flatnonzero(services)])

    def interpret_defenses(self, defenses=None):
        if defenses is None:
            defenses = self.defense_state
        return list(np.array(self.g.defense_names)[np.flatnonzero(defenses)])

    def interpret_attacks(self, attacks=None):
        if attacks is None:
            attacks = self.attack_state
        return list(np.array(self.g.attack_names)[np.flatnonzero(attacks)])

    def interpret_observation(self, observation):

        defenses = observation[: self.g.num_defenses]
        attacks = observation[self.g.num_defenses :]
        return self.interpret_defenses(defenses), self.interpret_attacks(attacks)

    def interpret_action(self, action):
        return (
            self.NO_ACTION_STR
            if action == self.NO_ACTION
            else self.g.defense_names[action - 1]
            if 0 < action <= self.g.num_defenses
            else "invalid action"
        )

    def generate_noise(self):
        """Generates a "noise" mask to use for false positives and
        negatives."""
        return self.rng.uniform(0, 1, self.num_attack_steps)

    def observe(self):
        """Observation of attack steps is subject to the true/false positive
        rates of an assumed underlying intrusion detection system Depending on
        the true and false positive rates for each step, ongoing attacks may
        not be reported, or non-existing attacks may be spuriously reported."""
        probabilities = self.noise
        false_negatives = self.attack_state & (probabilities >= self.false_negative)
        false_positives = (1 - self.attack_state) & (probabilities <= self.false_positive)
        detected = false_negatives | false_positives
        return np.append(self.defense_state, detected)

    def current_attack_step(self):
        """Returns the attack step the attacker is currently targeting."""
        current_step = (
            None
            if self.attack_index is None
            else self.NO_ACTION
            if self.attack_index == -1
            else self.g.attack_names[self.attack_index]
        )
        ttc_remaining = (
            -1
            if self.attack_index is None or self.attack_index == -1
            else self.ttc_remaining[self.attack_index]
        )
        return current_step, ttc_remaining

    @property
    def compromised_steps(self):
        return self.interpret_attacks()

    @property
    def compromised_flags(self):
        return [step_name for step_name in self.compromised_steps if "flag" in step_name]
