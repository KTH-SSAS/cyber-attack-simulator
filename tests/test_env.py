import dataclasses

import numpy as np
import pytest

from attack_simulator.config import EnvConfig
from attack_simulator.env import AttackSimulationEnv
from attack_simulator.graph import AttackGraph
from attack_simulator.renderer import AttackSimulationRenderer
from attack_simulator.sim import AttackSimulator


def test_env_spaces(env: AttackSimulationEnv, attack_graph: AttackGraph) -> None:
    num_defenses = attack_graph.num_defenses
    num_actions = env.action_space.n
    assert num_actions == num_defenses + 1
    num_attacks = attack_graph.num_attacks
    dim_observations = env.observation_space.shape[0]
    assert dim_observations == num_defenses + num_attacks


def test_env_seed(env: AttackSimulationEnv) -> None:
    assert env.seed() is not None
    assert [42] == env.seed()


def test_env_reset(env: AttackSimulationEnv) -> None:
    obs = np.array(env.reset())
    assert all(env.sim.observe() == obs)


@pytest.mark.parametrize("save_graphs", [False, True])
@pytest.mark.parametrize("save_logs", [False, True])
def test_env_render_save_graphs(save_graphs, save_logs, env_config, tmpdir) -> None:
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
