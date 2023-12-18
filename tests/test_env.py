import dataclasses

import numpy as np
import pytest
import torch
from attack_simulator.constants import AGENT_ATTACKER, AGENT_DEFENDER
from attack_simulator.env.env import AttackSimulationEnv
from attack_simulator.utils.config import EnvConfig
import attack_simulator
import attack_simulator.wrappers

def test_env_reset(env: AttackSimulationEnv) -> None:
    obs = np.array(env.reset())


def test_check_spaces(env: AttackSimulationEnv) -> None:
    attacker_obs_space = env.observation_space.spaces[AGENT_ATTACKER]
    defender_obs_space = env.observation_space.spaces[AGENT_DEFENDER]

    def check_space(space, obs):
        for k, v in obs.items():
            assert k in space.spaces, f"{k} not in {space.spaces}"
            assert space.spaces[k].contains(v), f"{k} {v} not in {space.spaces[k]}"

        assert space.contains(obs)
    
    # check that the observation space is valid
    obs, _ = env.reset()

    attacker_obs = obs[AGENT_ATTACKER]
    defender_obs = obs[AGENT_DEFENDER]

    check_space(attacker_obs_space, attacker_obs)
    check_space(defender_obs_space, defender_obs)

    assert env.observation_space.contains(obs)

    obs, *_ = env.step({AGENT_ATTACKER: (0, 0), AGENT_DEFENDER: (0, 0)})

    # check that the observation space is still valid after a step
    attacker_obs = obs[AGENT_ATTACKER]
    defender_obs = obs[AGENT_DEFENDER]

    check_space(attacker_obs_space, attacker_obs)
    check_space(defender_obs_space, defender_obs)

    assert env.observation_space.contains(obs)

    defender_obs_space.sample()
    attacker_obs_space.sample()
    


def test_env_step(env: AttackSimulationEnv) -> None:
    obs, info = env.reset()
    action = env.action_space.sample({"defender":info["defender"]["action_mask"], "attacker":info["attacker"]["action_mask"]})
    obs, reward, terminated, truncated, info = env.step(action)
    assert env._agent_ids
    assert "attacker" in obs
    assert "defender" in obs


def test_env_multiple_steps(env: AttackSimulationEnv) -> None:
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample({"defender":info["defender"]["action_mask"], "attacker":info["attacker"]["action_mask"]})
        obs, reward, terminated, truncated, info = env.step(action)
        assert "attacker" in obs
        assert "defender" in obs
        if not env._agent_ids:
            break

def test_all_graphs_multiple_steps() -> None:
    available_graphs = attack_simulator.available_graphs()
    import gymnasium as gym
    attack_simulator.register_envs()


    for graph in available_graphs:
        env = gym.make("AttackerEnv-v0", graph_name=graph)
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample(info["action_mask"])
            obs, reward, terminated, truncated, info = env.step(action)
            #assert "attacker" in obs
            #assert "defender" in obs
            if terminated or truncated:
                break
            #    break

def test_gym_api() -> None:
    import gymnasium as gym
    import gymnasium.utils.env_checker as env_checker
    from attack_simulator.env.gym import DefenderEnv, AttackerEnv
    gym.register("DefenderEnv-v0", entry_point=DefenderEnv)
    env = gym.make("DefenderEnv-v0", render_mode="human")
    env_checker.check_env(env.unwrapped, skip_render_check=True)


    gym.register("AttackerEnv-v0", entry_point=AttackerEnv)
    env = gym.make("AttackerEnv-v0")
    env_checker.check_env(env.unwrapped, skip_render_check=True)


def test_pettingzoo_api() -> None:
    from attack_simulator import parallel_env, env
    from pettingzoo.test.api_test import api_test
    from pettingzoo.test import parallel_api_test

    p_env = parallel_env({})
    parallel_api_test(p_env, num_cycles=50)
    aec_env = env({})
    api_test(aec_env, num_cycles=50)

def test_wrappers() -> None:
    import gymnasium as gym
    attack_simulator.register_envs()

    env = gym.make("DefenderEnv-v0", graph_name="test_graph", render_mode="human")

    wrapped_env = attack_simulator.wrappers.GraphWrapper(env)

    obs, _ = wrapped_env.reset()

    assert isinstance(obs, gym.spaces.GraphInstance)
    assert obs.nodes.shape == (8,)

    wrapped_env = attack_simulator.wrappers.LabeledGraphWrapper(env)
    obs, _ = wrapped_env.reset()
    assert isinstance(obs, gym.spaces.GraphInstance)
    assert obs.nodes.shape == (8, 4)

    wrapped_env = attack_simulator.wrappers.BoxWrapper(env)
    obs, _ = wrapped_env.reset()
    assert obs.shape == (8,)


    wrapped_env = attack_simulator.wrappers.LabeledBoxWrapper(env)
    obs, _ = wrapped_env.reset()
    assert obs.shape == (8, 4)

    pass
