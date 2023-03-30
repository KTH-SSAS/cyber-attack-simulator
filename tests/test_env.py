import dataclasses

import numpy as np
import pytest

from attack_simulator.env.env import AttackSimulationEnv
from attack_simulator.render.renderer import AttackSimulationRenderer
from attack_simulator.utils.config import EnvConfig


def test_env_reset(env: AttackSimulationEnv) -> None:
    obs = np.array(env.reset())


def test_check_spaces(env: AttackSimulationEnv) -> None:
    
    obs, _ = env.reset()

    attacker_obs_space = env.observation_space.spaces[AGENT_ATTACKER]
    defender_obs_space = env.observation_space.spaces[AGENT_DEFENDER]

    attacker_obs = obs[AGENT_ATTACKER]
    defender_obs = obs[AGENT_DEFENDER]

    assert attacker_obs_space.contains(attacker_obs)
    assert defender_obs_space.contains(defender_obs)

    assert env.observation_space.contains(obs)

    obs, *_ = env.step({AGENT_ATTACKER: 0, AGENT_DEFENDER: 0})

    attacker_obs = obs[AGENT_ATTACKER]
    defender_obs = obs[AGENT_DEFENDER]

    assert attacker_obs_space.contains(attacker_obs)
    assert defender_obs_space.contains(defender_obs)

    assert env.observation_space.contains(obs)





def test_env_step(env: AttackSimulationEnv) -> None:
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated["__all__"] is False
    assert truncated["__all__"] is False
    assert "attacker" in obs
    assert "defender" in obs


def test_env_multiple_steps(env: AttackSimulationEnv) -> None:
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert "attacker" in obs
        assert "defender" in obs
        if terminated["__all__"] or truncated["__all__"]:
            break


@pytest.mark.skip(reason="needs to be updated to comply with gym standards")
@pytest.mark.parametrize("save_graphs", [False, True])
@pytest.mark.parametrize("save_logs", [False, True])
def test_env_render_save_graphs(
    save_graphs: bool, save_logs: bool, env_config: EnvConfig, tmpdir
) -> None:
    config: EnvConfig = dataclasses.replace(
        env_config, save_graphs=save_graphs, save_logs=save_logs
    )
    env = AttackSimulationEnv(config)
    frames = AttackSimulationRenderer.HTML.replace(".html", "_frames")
    with tmpdir.as_cwd():
        env.reset()
        env.render()
        if save_graphs or save_logs:
            assert env.renderer is not None

        render_dir = tmpdir.join("render")
        render_dir = render_dir.join("test")
        render_dir = render_dir.join("ep-0")

        if save_graphs or save_logs:
            files = render_dir.listdir()
            basenames = [f.basename for f in files]
            assert len(files) == int(save_graphs) + int(save_logs)
            assert (frames in basenames) == save_graphs
            assert (AttackSimulationRenderer.LOGS in basenames) == save_logs

        _, _, done, _ = env.step(0)  # no action
        assert not done
        env.render()
        _, _, done, _ = env.step(1)  # disable first defense
        assert not done
        env.render()
        _, _, done, _ = env.step(2)  # disable second defense --> terminate
        assert done
        env.render()

        if save_graphs or save_logs:
            files = render_dir.listdir()
            basenames = [f.basename for f in files]
            assert len(files) == 2 * int(save_graphs) + int(save_logs)
            assert (AttackSimulationRenderer.HTML in basenames) == save_graphs

            if save_graphs:
                files = render_dir.join(frames).listdir()
                assert len(files) == 5

            if save_logs:
                with open(render_dir.join(AttackSimulationRenderer.LOGS), encoding="utf8") as logs:
                    lines = logs.readlines()
                assert 4 <= len(lines)
