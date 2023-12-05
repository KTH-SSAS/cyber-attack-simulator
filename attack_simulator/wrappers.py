from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import RenderFrame

from gymnasium import Wrapper
from gymnasium import spaces
from gymnasium.spaces.graph import GraphInstance
import numpy as np

from attack_simulator.env.env import AttackSimulationEnv


class GraphWrapper(Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        self.observation_space = spaces.Graph(
            spaces.Discrete(2),
            spaces.Discrete(1),
        )

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.env.render()

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        graph_obs = self._to_graph(obs)
        return graph_obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        graph_obs = self._to_graph(obs)
        return graph_obs, reward, terminated, truncated, info

    @staticmethod
    def _to_graph(obs: dict[str, Any]) -> GraphInstance:
        return GraphInstance(obs["observation"], None, obs["edges"])


class LabeledGraphWrapper(Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        self.observation_space = spaces.Graph(
            spaces.Box(0, len(env.unwrapped.vocab), shape=(1, 4), dtype=np.int8),
            spaces.Discrete(1),
        )

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.env.render()

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._to_graph(obs), info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._to_graph(obs), reward, terminated, truncated, info

    @staticmethod
    def _to_graph(obs: dict[str, Any]) -> GraphInstance:
        nodes = np.concatenate(
            [obs["observation"], obs["asset"], obs["asset_id"], obs["step_name"]], axis=1
        )
        return GraphInstance(nodes, None, obs["edges"])
