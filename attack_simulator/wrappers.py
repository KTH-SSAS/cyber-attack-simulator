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
        self.defense_steps = None

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.env.render()

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        defense_steps = set(np.flatnonzero(info["action_mask"][1]))
        graph_obs = self._to_graph(obs, defense_steps)
        self.defense_steps = defense_steps
        return graph_obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        graph_obs = self._to_graph(obs, self.defense_steps)
        return graph_obs, reward, terminated, truncated, info

    @staticmethod
    def _to_graph(obs: dict[str, Any], defense_steps: set = None) -> GraphInstance:
        edges = obs["edges"]

        # Add reverse edges from the defense steps children to the defense steps
        # themselves
        if defense_steps is not None:
            for p, c in zip(edges[0, :], edges[1, :]):
                if p in defense_steps:
                    new_edge = np.array([c, p]).reshape((2, 1))
                    edges = np.concatenate((edges, new_edge), axis=1)


        return GraphInstance(obs["observation"], None, edges)


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
        return self._to_graph(obs), info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._to_graph(obs), reward, terminated, truncated, info

    @staticmethod
    def _to_graph(obs: dict[str, Any]) -> GraphInstance:
        nodes = np.stack(
            [obs["observation"], obs["asset"], obs["asset_id"], obs["step_name"]]
        ).T
        return GraphInstance(nodes, None, obs["edges"])
