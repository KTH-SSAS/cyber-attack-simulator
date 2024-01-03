from typing import Any, List, Tuple
from attack_simulator.agents.agent import RandomAgent, RandomActiveAgent, NothingAgent
from attack_simulator.env.attacksimulator import env, raw_env, parallel_env
from attack_simulator.utils.config import EnvConfig, SimulatorConfig
from attack_simulator.env.gym import DefenderEnv, AttackerEnv
from attack_simulator.env.env import obs_to_graphviz
from attack_simulator.examplemanager import available_graphs, show_graph
import gymnasium as gym
from collections import defaultdict
import numpy as np

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

def get_baselines_for_env(env: gym.Env, episodes: int=100) -> dict:

    agents = {
        "random" : RandomAgent({}),
        "random_without_wait" : RandomActiveAgent({}), 
        "nop" :NothingAgent({})
    }

    results = defaultdict(list)
    
    for name, agent in agents.items():
        for _ in range(episodes):
            o, i = env.reset()
            done = False
            ret = 0
            while not done:
                o["action_mask"] = i["action_mask"]
                o, r, term, trunc, i = env.step(agent.compute_action_from_dict(o, i["action_mask"]))
                ret += r
                done = term or trunc

            results[name].append(ret)

    final_results = {}
    for name, rets in results.items():
        final_results[name] = {"mean": np.mean(rets), "std": np.std(rets)}
        
    return final_results

__all__ = [
    "obs_to_graphviz",
    "env",
    "raw_env",
    "parallel_env",
    "EnvConfig",
    "SimulatorConfig",
    "available_graphs",
    "show_graph",
]