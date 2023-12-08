from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import RenderFrame

from gymnasium import Wrapper
from gymnasium import spaces
from gymnasium.spaces.graph import GraphInstance
import numpy as np

from attack_simulator.env.env import AttackSimulationEnv

BIG_INT = 2**63 - 2

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
        info["action_mask"] = obs["action_mask"]
        return graph_obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        graph_obs = self._to_graph(obs)
        info["action_mask"] = obs["action_mask"]
        return graph_obs, reward, terminated, truncated, info

    @staticmethod
    def _to_graph(obs: dict[str, Any]) -> GraphInstance:
        return GraphInstance(obs["observation"], None, obs["edges"])


class LabeledGraphWrapper(Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        self.observation_space = spaces.Graph(
            spaces.Box(0, BIG_INT, shape=(4,), dtype=np.int64),
            spaces.Discrete(1),
        )

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.env.render()

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        info["action_mask"] = obs["action_mask"]
        return self._to_graph(obs), info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["action_mask"] = obs["action_mask"]
        return self._to_graph(obs), reward, terminated, truncated, info

    @staticmethod
    def _to_graph(obs: dict[str, Any]) -> GraphInstance:
        nodes = np.stack(
            [obs["observation"], obs["asset"], obs["asset_id"], obs["step_name"]]
        ).T
        return GraphInstance(nodes, None, obs["edges"])
