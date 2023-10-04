import dataclasses

import numpy as np
import pytest
import torch
from attack_simulator.constants import AGENT_ATTACKER, AGENT_DEFENDER
from attack_simulator.env.env import AttackSimulationEnv
from attack_simulator.utils.config import EnvConfig


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
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated["__all__"] is False
    assert truncated["__all__"] is False
    assert "attacker" in obs
    assert "defender" in obs


def test_env_multiple_steps(env: AttackSimulationEnv) -> None:
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert "attacker" in obs
        assert "defender" in obs
        if terminated["__all__"] or truncated["__all__"]:
            break


@pytest.mark.parametrize("agent", ["attacker", "defender"])
def test_env_multiple_steps_single(env: AttackSimulationEnv, agent) -> None:
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        del action[agent]
        obs, reward, terminated, truncated, info = env.step(action)
        # note this techically is not how multi agents envs work in rllib
        # only agents that took an action should be in the obs
        assert "attacker" in obs
        assert "defender" in obs
        if terminated["__all__"] or truncated["__all__"]:
            break

