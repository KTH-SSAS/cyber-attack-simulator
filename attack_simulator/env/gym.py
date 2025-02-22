import functools
from typing import Any, Dict, SupportsFloat, List

import gymnasium as gym
from gymnasium.core import RenderFrame
import gymnasium.utils.env_checker as env_checker
from attack_simulator.agents.agent import Agent
import attack_simulator.agents.attackers.searchers as searchers

from attack_simulator.constants import AGENT_ATTACKER, AGENT_DEFENDER

from attack_simulator.env.env import AttackSimulationEnv
from attack_simulator.utils.config import EnvConfig
import numpy as np

class AttackerEnv(gym.Env):

    metadata = AttackSimulationEnv.metadata

    def __init__(self, **kwargs: Any) -> None:
        config = {
            "graph_name": kwargs.get("graph_name", "four_ways"),
            "sim_false_negative_rate": 0.0,
            "sim_false_positive_rate": 0.0,
            "vocab_filename": kwargs.get("vocab_file", None),
        }
        self.env = AttackSimulationEnv(EnvConfig.from_dict(config))
        self.observation_space = self.env.observation_space[AGENT_ATTACKER]
        self.action_space = self.env.action_space[AGENT_ATTACKER]
        self.render_mode = kwargs.get("render_mode", None)
        self.env.render_mode = self.render_mode
        super().__init__()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        obs, info = self.env.reset(seed=seed, options=options)
        return obs[AGENT_ATTACKER], info[AGENT_ATTACKER]

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs: Dict[str, Any]
        obs, rewards, terminated, truncated, infos = self.env.step({AGENT_ATTACKER: action})
        return (
            obs[AGENT_ATTACKER],
            rewards[AGENT_ATTACKER],
            terminated[AGENT_ATTACKER],
            truncated[AGENT_ATTACKER],
            infos[AGENT_ATTACKER],
        )

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.env.render()
    
    @property
    def vocab(self) -> Dict[str, int]:
        return self.env.vocab
    
    @property
    def reverse_vocab(self) -> List[str]:
        return self.env.reverse_vocab


class DefenderEnv(gym.Env):
    
    metadata = AttackSimulationEnv.metadata
    attacker: Agent
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        undirected_defenses = kwargs.get("undirected_defenses", False)
        config = {
            "graph_name": kwargs.get("graph_name", "four_ways"),
            "sim_false_negative_rate": kwargs.get("false_negative_rate", 0.0),
            "sim_false_positive_rate": kwargs.get("false_positive_rate", 0.0),
            "undirected_defenses": undirected_defenses,
            "vocab_filename": kwargs.get("vocab_file", None),
        }
        self.env = AttackSimulationEnv(EnvConfig.from_dict(config))
        attacker_class: str = kwargs.get("attacker_class", "BreadthFirstAttacker")
        self.attacker_class = searchers.agents[attacker_class]
        self.observation_space = self.env.observation_space[AGENT_DEFENDER]
        self.action_space = self.env.action_space[AGENT_DEFENDER]
        self.randomize = kwargs.get("randomize_attacker_behavior", False)
        self.render_mode = kwargs.get("render_mode", None)
        
        self.env.render_mode = self.render_mode

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.attacker = self.attacker_class({"seed": seed, "randomize": self.randomize})
        obs, info = self.env.reset(seed=seed, options=options)
        self.attacker_obs = obs[AGENT_ATTACKER]
        self.attacker_mask = info[AGENT_ATTACKER]["action_mask"]
        return obs[AGENT_DEFENDER], info[AGENT_DEFENDER]

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        attacker_action = self.attacker.compute_action_from_dict(self.attacker_obs, self.attacker_mask)
        obs: Dict[str, Any]
        obs, rewards, terminated, truncated, infos = self.env.step(
            {AGENT_DEFENDER: action, AGENT_ATTACKER: attacker_action}
        )
        self.attacker_obs = obs[AGENT_ATTACKER]
        self.attacker_mask = infos[AGENT_ATTACKER]["action_mask"]
        return (
            obs[AGENT_DEFENDER],
            rewards[AGENT_DEFENDER],
            terminated[AGENT_DEFENDER],
            truncated[AGENT_DEFENDER],
            infos[AGENT_DEFENDER],
        )

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.env.render()
    
    @property
    def vocab(self) -> Dict[str, int]:
        return self.env.vocab
    
    @property
    def reverse_vocab(self) -> List[str]:
        return self.env.reverse_vocab
    
    @staticmethod
    def add_reverse_edges(edges: np.ndarray, defense_steps: set) -> np.ndarray:
        # Add reverse edges from the defense steps children to the defense steps
        # themselves
        if defense_steps is not None:
            for p, c in zip(edges[0, :], edges[1, :]):
                if p in defense_steps:
                    new_edge = np.array([c, p]).reshape((2, 1))
                    edges = np.concatenate((edges, new_edge), axis=1)
        return edges


if __name__ == "__main__":
    gym.register("DefenderEnv-v0", entry_point=DefenderEnv)
    env = gym.make("DefenderEnv-v0", render_mode="human")
    env_checker.check_env(env.unwrapped)


    gym.register("AttackerEnv-v0", entry_point=AttackerEnv)
    env = gym.make("AttackerEnv-v0")
    env_checker.check_env(env.unwrapped)
