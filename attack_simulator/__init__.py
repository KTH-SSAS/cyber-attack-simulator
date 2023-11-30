from typing import Any, List, Tuple
from attack_simulator.env.attacksimulator import env, raw_env, parallel_env
from attack_simulator.utils.config import EnvConfig, SimulatorConfig
from attack_simulator.env.gym import DefenderEnv, AttackerEnv
from .examplemanager import available_graphs, show_graph
import gymnasium as gym


def register_envs() -> Tuple[str, str]:
    defend_env_name = "DefenderEnv-v0"
    attack_env_name = "AttackerEnv-v0"
    gym.register(defend_env_name, entry_point=DefenderEnv)
    gym.register(attack_env_name, entry_point=AttackerEnv)
    return defend_env_name, attack_env_name


def defender_gym(**kwargs: Any) -> gym.Env:
    return gym.make("DefenderEnv-v0", **kwargs)


def attacker_gym(**kwargs: Any) -> gym.Env:
    return gym.make("AttackerEnv-v0", **kwargs)


def list_available_attackers() -> List[str]:
    return ["BreadthFirstAttacker", "DepthFirstAttacker"]


__all__ = [
    "env",
    "raw_env",
    "parallel_env",
    "EnvConfig",
    "SimulatorConfig",
    "available_graphs",
    "show_graph",
]
