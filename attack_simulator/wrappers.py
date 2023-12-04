
from typing import Any

import gymnasium as gym
from gymnasium.core import RenderFrame

from gymnasium import Wrapper
from gymnasium import spaces
from gymnasium.spaces.graph import GraphInstance
import numpy as np

from attack_simulator.env.env import AttackSimulationEnv

class GraphWrapper(Wrapper):

    def __init__(self, env: AttackSimulationEnv) -> None:
        super().__init__(env)

        self.observation_space = spaces.Graph(
            spaces.Discrete(2),
            spaces.Discrete(1),
        )

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._graph.render()
    

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        graph_obs = GraphInstance(obs["observation"], None, obs["edges"])
        return graph_obs, info
    
class LabeledGraphWrapper(Wrapper):

    def __init__(self, env: AttackSimulationEnv) -> None:
        super().__init__(env)

        self.observation_space = spaces.Graph(
            spaces.Box(0, len(env.vocab), shape=(1, 4), dtype=np.int8),
            spaces.Discrete(1),
        )

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._graph.render()
    

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)

        full_nodes = np.concatenate([obs["observation"], obs["asset"], obs["asset_id"], obs["step_name"]], axis=1)

        graph_obs = GraphInstance(full_nodes, None, obs["edges"])
        return graph_obs, info


