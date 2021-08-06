"""Simple agents that can be used as baselines for performance"""
import numpy as np

from ..rng import get_rng
from .agent import Agent


class RandomAgent(Agent):
    """Agent that will pick a random action each turn."""

    def __init__(self, agent_config):
        self.rng, _ = get_rng(agent_config.get("random_seed"))
        self.num_actions = agent_config["num_actions"]

    def act(self, observation=None):
        return self.rng.integers(self.num_actions)


class SkipAgent(Agent):
    """Agent that will always select no action, i.e. does nothing (or "skips"), on each turn."""

    def __init__(self, agent_config=None):
        pass

    def act(self, observation=None):
        return 0


class DisableProbabilityAgent(Agent):
    """An agent that mixes `no action` and `random action` with a given `disable_probability`."""

    def __init__(self, agent_config):
        self.rng, _ = get_rng(agent_config.get("random_seed"))
        self.num_services = agent_config["num_actions"] - 1
        self.disable_probability = agent_config.get("disable_probability", 0.1)

    def act(self, observation):
        enabled_services = np.array(observation[-self.num_services :])
        disable = self.rng.uniform(0, 1) < self.disable_probability
        return self.rng.choice(np.flatnonzero(enabled_services)) + 1 if disable else 0


class RuleBasedAgent(Agent):
    """
    Rule-based agent following a simple IF-THIS-THEN-THAT rule.

    IF the attacker seems to have compromised an attack step preceeding valuable step,
    THEN disable the corresponding service.
    """

    def __init__(self, agent_config):
        self.g = agent_config["attack_graph"]
        self.attack_state = np.full(self.g.num_attacks, 0)

    def act(self, observation):
        service_state = np.array(observation[: self.g.num_services])
        attack_state = np.array(observation[self.g.num_services :])

        changed = (1 - self.attack_state) & attack_state
        self.attack_state = attack_state

        action = 0  # no action

        if any(changed):
            # pick the change first change, there _should_ be only one
            attack_name = self.g.attack_names[np.flatnonzero(changed)[0]]
            attack_step = self.g.attack_steps[attack_name]

            # If an attack step has been compromised which has a valuable child
            # then disable the corresponding service.
            if any([self.g.attack_steps[child].reward > 0 for child in attack_step.children]):
                service_name = attack_step.asset
                if attack_step.service:
                    service_name += "." + attack_step.service
                service_index = self.g.service_names.index(service_name)
                if service_state[service_index]:
                    action = service_index + 1  # + 1 because action == 0 is no action.
        return action
