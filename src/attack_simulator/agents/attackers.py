"""
This module implements Attacker agents

Note that the observation space for attackers is the current attack surface,
and their action space is 0 for "no action" or 1 + the index of an attack in
the current attack surface; essentially [0, num_attacks] inclusive.
"""
import logging

import numpy as np

from ..rng import get_rng
from .agent import Agent

logger = logging.getLogger("simulator")


class RandomAttacker(Agent):
    def __init__(self, agent_config):
        self.rng, _ = get_rng(agent_config.get("random_seed"))

    def act(self, observation):
        valid_attack_indices = np.flatnonzero(observation)
        return self.rng.choice(valid_attack_indices) + 1


class RandomNoActionAttacker(Agent):
    def __init__(self, agent_config):
        self.rng, _ = get_rng(agent_config.get("random_seed"))

    def act(self, observation):
        valid_attacks = np.concatenate([[0], np.flatnonzero(observation) + 1])
        return self.rng.choice(valid_attacks)


class RoundRobinAttacker(Agent):
    def __init__(self, agent_config=None):
        self.last = 0

    def act(self, observation):
        valid = np.flatnonzero(observation)
        above = valid[self.last < valid]
        self.last = valid[0] if 0 == above.size else above[0]
        return self.last + 1


class RoundRobinNoActionAttacker(Agent):
    def __init__(self, agent_config=None):
        self.last = 0

    def act(self, observation):
        valid = np.concatenate([[0], np.flatnonzero(observation) + 1])
        above = valid[self.last < valid]
        self.last = valid[0] if 0 == above.size else above[0]
        return self.last


class WellInformedAttacker(Agent):
    """An Attacker with complete information on the underlying AttackGraph"""

    def __init__(self, agent_config):
        graph = agent_config["attack_graph"]
        steps = graph.attack_steps
        names = graph.attack_names
        self._ttc = dict(zip(names, agent_config["ttc"]))
        self._rewards = dict(zip(names, agent_config["rewards"]))
        values = {}
        total = self._value(steps, values, graph.root)
        self._value = lambda: (_ for _ in ()).throw(RuntimeError("called disabled method"))
        logger.info(f"{self.__class__.__name__}: total discounted value: {total}")
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
