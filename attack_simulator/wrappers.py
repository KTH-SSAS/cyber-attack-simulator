from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import Env, RenderFrame

from gymnasium import Wrapper
from gymnasium import spaces
from gymnasium.spaces.graph import GraphInstance
import numpy as np

from attack_simulator.env.env import AttackSimulationEnv
from attack_simulator.env.gym import DefenderEnv

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
        nodes = np.stack([obs["observation"], obs["asset"], obs["asset_id"], obs["step_name"]]).T
        return GraphInstance(nodes, None, obs["edges"])


class BoxWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = self.env.observation_space["observation"]

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return super().render()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs = obs["observation"]
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs["observation"]
        return obs, reward, terminated, truncated, info


class LabeledBoxWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        num_nodes = self.env.observation_space["observation"].shape[0]
        self.observation_space = spaces.Box(0, BIG_INT, shape=(num_nodes, 4), dtype=np.int64)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return super().render()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.stack([obs["observation"], obs["asset"], obs["asset_id"], obs["step_name"]]).T
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.stack([obs["observation"], obs["asset"], obs["asset_id"], obs["step_name"]]).T
        return obs, reward, terminated, truncated, info


def _to_binary(val, max_val):
    return np.array(list(np.binary_repr(val, width=max_val.bit_length())), dtype=np.int64)

def vec_to_binary(vec, max_val):
    return np.array([_to_binary(val, max_val) for val in vec])

class BinaryEncodingWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self._og_observation_space = self.env.observation_space
        self.observation_space = spaces.Dict(
            {
                "nodes": spaces.Box(
                    0,
                    1,
                    shape=(self.env.observation_space["observation"].shape[0], 4),
                    dtype=np.int64,
                ),
                "edges": self.env.observation_space["edges"],
            }
        )

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return super().render()

    def _convert(self, obs: dict[str, Any]) -> dict[str, Any]:
        num_assets = len(self.env.unwrapped.vocab)
        num_steps = len(self.env.unwrapped.vocab)
        edges = obs["edges"]
        obs = np.concatenate(
            [
                vec_to_binary(obs["observation"], 3),
                vec_to_binary(obs["asset"], num_assets),
                vec_to_binary(obs["step_name"], num_steps),
            ]
            , axis=1
        )
        obs = {
            "nodes": obs,
            "edges": edges,
        }
        return obs
    
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        obs = self._convert(obs)
        
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._convert(obs)
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    gym.register("DefenderEnv-v0", entry_point=DefenderEnv)
    env = gym.make("DefenderEnv-v0", render_mode="human")
    env = BinaryEncodingWrapper(env)

    obs = env.reset()
    pass