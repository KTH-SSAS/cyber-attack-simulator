from copy import deepcopy
from typing import Tuple

import gym
import numpy as np

from .env import AttackSimulationEnv


class AttackSimulationAlphaZeroEnv(AttackSimulationEnv):
    """Adapt AttackSimulationEnv for Ray's `contrib/AlphaZero` implementation.

    NOTE: 'contrib/AlphaZero' does NOT appear to work without its custom dense model

     from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
     rllib.models.ModelCatalog.register_custom_model("alpha_zero_dense_model", DenseModel)

     trainer_config["model"]["custom_model"] = "alpha_zero_dense_model"
    """

    def __init__(self, config: dict):
        super().__init__(config)
        shape = (self.action_space.n,)
        self.observation_space = gym.spaces.Dict(
            dict(obs=self.observation_space, action_mask=gym.spaces.Box(0, 1, shape, dtype="int8"))
        )
        self.mask = np.full(shape, 1, dtype="int8")
        self.total_reward = 0.0

    def reset(self) -> dict:  # type: ignore
        self.total_reward = 0
        observation = super().reset()
        return dict(obs=observation, action_mask=self.mask)

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:  # type: ignore
        observation, reward, done, info = super().step(action)
        self.total_reward += reward
        reward = self.total_reward if done else 0
        return dict(obs=observation, action_mask=self.mask), reward, done, info

    def get_state(self) -> tuple:
        state = (
            self.simulation_time,
            self.ttc_remaining,
            self.attack_surface,
            self.attack_state,
            self.service_state,
            self._observation,
            self.rng,
            self.attacker,
            self.total_reward,
        )
        return deepcopy(state)

    def set_state(self, state: tuple) -> dict:
        (
            self.simulation_time,
            self.ttc_remaining,
            self.attack_surface,
            self.attack_state,
            self.service_state,
            self._observation,
            self.rng,
            self.attacker,
            self.total_reward,
        ) = deepcopy(state)
        return dict(obs=self.observation, action_mask=self.mask)
