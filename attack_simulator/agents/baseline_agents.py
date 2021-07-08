"""Simple agents that can be used as baselines for performance"""
import numpy as np
import torch

class RandomMCAgent():

    def __init__(self, num_actions, allow_skip=True) -> None:
        self.num_actions = num_actions + 1 if allow_skip else num_actions
        self.can_skip = allow_skip

    def act(self, *args):
        return np.random.randint(0, self.num_actions)

    def update(self, *args):
        return 0

    def calculate_loss(self, *args, **kwargs):
        return torch.Tensor([0])

    def eval(self):
        ...

    def train(self):
        ...

class SkipAgent():

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