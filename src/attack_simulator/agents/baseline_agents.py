"""Simple agents that can be used as baselines for performance"""
import numpy as np
import torch

from .tabular_agents import Agent


class RandomMCAgent:
    """Agent that will pick a random action each turn. Returns random loss."""

    def __init__(self, num_actions, allow_skip=True) -> None:
        self.num_actions = num_actions + 1 if allow_skip else num_actions
        self.can_skip = allow_skip

    def act(self, *args):
        return np.random.randint(0, self.num_actions)

    def update(self, *args):
        return self.calculate_loss()

    def calculate_loss(self, *args, **kwargs):
        return torch.Tensor([np.random.rand()])

    def eval(self):
        ...

    def train(self):
        ...


class SkipAgent:
    """Agent that will always skip, i.e. do nothing, each turn."""

    def __init__(self) -> None:
        self.can_skip = True

    def act(self, *args):
        return 0

    def update(self, *args):
        return 0

    def calculate_loss(self, *args, **kwargs):
        return torch.Tensor([0])

    def eval(self):
        ...

    def train(self):
        ...


class RuleBasedAgent(Agent):
    """
    Rule-based agent following a simple IF-THIS-THEN-THAT rule.

    IF the attacker seems to have compromised an attack step preceeding valuable step,
    THEN disable the corresponding service.
    """

    def __init__(self, env) -> None:
        self.attack_graph = env.attack_graph
        self.attacker = env.attacker
        self.can_skip = True
        self.n_action = 0
        self.previous_state = [False] * len(self.attack_graph.attack_steps)

    def act(self, state):
        action_id = 0
        # When a step with a valuable child got compromised, disable the corresponding service.
        for step_id in range(0, len(state)):
            if state[step_id]:
                step_name = self.attack_graph.attack_names[step_id]
                for child_name in self.attack_graph.attack_steps[step_name].children:
                    if self.env.rewards[child_name] > 0:
                        service = self.corresponding_service(step_name)
                        if service in self.env.enabled_services:
                            # action_id + 1 because action == 0 is no action.
                            action_id = self.attack_graph.service_names.index(service) + 1
        # If no service should be disabled, then return 0
        self.previous_state = state
        return action_id

    def corresponding_service(self, attack_step_name):
        step = self.attack_graph.attack_steps[attack_step_name]
        service_name = step.asset
        if step.service:
            service_name += "." + step.service
        return service_name

    def update(self, rewards):
        return torch.Tensor([0])

    def calculate_loss(self, rewards, normalize_returns=False):
        return torch.Tensor([0])

    def eval(self):
        pass

    def train(self):
        pass
