"""This module implements Attacker agents.

Note that the observation space for attackers is the current attack
surface, and their action space is 0 for "no action" or 1 + the index of
an attack in the current attack surface; essentially [0, num_attacks]
inclusive.
"""
import logging
from typing import Any, Dict

import numpy as np

from ...constants import UINT
from ...utils.rng import get_rng
from ..agent import Agent, RandomActiveAgent

logger = logging.getLogger("simulator")


class RandomAttacker(RandomActiveAgent):
    ...


class RandomNoActionAttacker(Agent):
    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        self.rng, _ = get_rng(agent_config.get("random_seed"))

    def act(self, observation: np.ndarray) -> int:
        attack_surface = observation[0]
        valid_attacks = np.concatenate([[0], np.flatnonzero(attack_surface)])
        return self.rng.choice(valid_attacks) + observation["action_offset"]


class RoundRobinAttacker(Agent):
    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        self.last = 0

    def compute_action_from_dict(self, observation: Dict[str, Any]) -> UINT:
        attack_surface = observation["action_mask"].reshape(-1)[observation["action_offset"] :]
        valid = np.flatnonzero(attack_surface)
        above = valid[self.last < valid]
        self.last = valid[0] if 0 == above.size else above[0]
        return self.last + observation["action_offset"]


class RoundRobinNoActionAttacker(Agent):
    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        self.last = 0

    def act(self, observation: np.ndarray) -> int:
        valid = np.concatenate([[0], np.flatnonzero(observation) + 1])
        above = valid[self.last < valid]
        self.last = valid[0] if 0 == above.size else above[0]
        return self.last + observation["action_offset"]


class WellInformedAttacker(Agent):
    """An Attacker with complete information on the underlying AttackGraph."""

    def __init__(self, agent_config: dict) -> None:
        super().__init__(agent_config)
        graph = agent_config["attack_graph"]
        steps = graph.attack_steps
        names = graph.attack_names
        # self._ttc = dict(zip(names, graph.ttc_params))
        self._rewards = dict(zip(names, graph.reward_params))
        values: dict = {}
        """
        self._value = (
        lambda x, y, z: (_ for _ in ()).throw(RuntimeError("called disabled method"))
        )
        """
        total = self._value(steps, values, graph.root)
        logger.info("%s: total discounted value: %d", self.__class__.__name__, total)
        # del self._ttc
        del self._rewards

        self.attack_values = np.array([values[name] for name in names])

    def _value(
        self, attack_steps: dict, attack_values: dict, attack_name: str, discount_rate: float = 0.1
    ) -> dict:
        """Recursively compute the value of each attack step.

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

    def act(self, observation: np.ndarray) -> int:
        """Selecting the attack step with the highest net present value."""
        return int(np.argmax(self.attack_values * observation)) + 1


class InformedAttacker(WellInformedAttacker):
    """An Attacker with access to the AttackGraph **without sampled TTC and
    rewards**"""

    def __init__(self, agent_config: dict):
        # graph: AttackGraph = agent_config["attack_graph"]
        # replace per-episode sampled info with base parameters
        super().__init__(agent_config)
