import dataclasses

import numpy as np
import pytest

from attack_simulator.config import EnvConfig
from attack_simulator.env import AttackSimulationEnv
from attack_simulator.graph import AttackGraph
from attack_simulator.renderer import AttackSimulationRenderer
from attack_simulator.sim import AttackSimulator


def test_env_spaces(env, attack_graph: AttackGraph):
    num_defenses = attack_graph.num_defenses
    num_actions = env.action_space.n
    assert num_actions == num_defenses + 1
    num_attacks = attack_graph.num_attacks
    dim_observations = env.observation_space.shape[0]
    assert dim_observations == num_defenses + num_attacks


def test_env_seed(env: AttackSimulationEnv):
    assert env.seed() is not None
    assert [42] == env.seed()


def test_env_reset(env: AttackSimulationEnv):
    obs = np.array(env.reset())
    assert all(env.sim.observe() == obs)


def test_env_first_observation(simulator: AttackSimulator):
    obs = np.array(simulator.observe())
    assert all(obs == [1] * simulator.g.num_defenses + [0] * simulator.g.num_attacks)


def test_sim_noisy_observation(simulator: AttackSimulator):

    # Set observations to be very noisy
    simulator.false_negative = 0.5
    simulator.false_positive = 0.5

    obs1 = simulator.observe()
    obs2 = simulator.observe()

    # If not time passes, the observations do not change
    assert all(obs1 == obs2)

    # When time changes, so will the observation
    simulator.step()

    obs2 = simulator.observe()

    assert any(obs1 != obs2)


def test_env_interpretations(simulator: AttackSimulator):
    obs = simulator.observe()
    i_obs = simulator.interpret_observation(obs)
    i_def = simulator.interpret_defenses(simulator.defense_state)
    i_att = simulator.interpret_attacks(simulator.attack_state)
    assert i_obs == (i_def, i_att)


def test_env_action(simulator: AttackSimulator, test_defense_steps):
    assert simulator.interpret_action(0) == simulator.NO_ACTION_STR
    for i, service in enumerate(test_defense_steps):
        assert simulator.interpret_action(i + 1) == service


def test_env_action_probs(env: AttackSimulationEnv, test_defense_steps):
    probs = [1] * env.num_actions
    i_probs = env.interpret_action_probabilities(probs)
    assert all(map(lambda v: v == 1, i_probs.values()))
    assert list(i_probs) == [env.NO_ACTION] + test_defense_steps


def test_env_render(env: AttackSimulationEnv, tmpdir):
    with tmpdir.as_cwd():
        assert env.render() is True


def test_env_default_config(env_config):
    env = AttackSimulationEnv(env_config)
    # defaults for project graph
    num_defenses = 2
    num_attacks = 6
    assert all(np.array(env.reset()) == [1] * num_defenses + [0] * num_attacks)


@pytest.mark.parametrize("save_graphs", [False, True])
@pytest.mark.parametrize("save_logs", [False, True])
def test_env_render_save_graphs(save_graphs, save_logs, env_config, tmpdir):
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
