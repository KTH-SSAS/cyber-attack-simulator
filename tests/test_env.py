import numpy as np
import pytest

from attack_simulator.agents import ATTACKERS
from attack_simulator.env import AttackSimulationEnv


def test_env_spaces(test_env, test_graph):
    num_services = test_graph.num_services
    num_actions = test_env.action_space.n
    assert num_actions == num_services + 1
    num_attacks = test_graph.num_attacks
    dim_observations = len(test_env.observation_space.spaces)
    assert dim_observations == num_services + num_attacks


def test_env_seed(test_env):
    assert test_env.seed() is not None
    assert 42 == test_env.seed(42)


def test_env_reset(test_env):
    obs = np.array(test_env.reset())
    assert all(test_env.observation == obs)


def test_env_first_observation(test_env):
    obs = np.array(test_env.observation)
    assert all(obs == [1] * test_env.g.num_services + [0] * test_env.g.num_attacks)


def test_env_interpretations(test_env):
    i_obs = test_env._interpret_observation()
    i_srv = test_env._interpret_services()
    i_att = test_env._interpret_attacks()
    assert i_obs == (i_srv, i_att)


def test_env_action(test_env, test_services):
    assert test_env._interpret_action(0) == test_env.NO_ACTION
    for i, service in enumerate(test_services):
        assert test_env._interpret_action(i + 1) == service


def test_env_action_probs(test_env, test_services):
    probs = [1] * test_env.num_actions
    i_probs = test_env._interpret_action_probabilities(probs)
    assert all(map(lambda v: v == 1, i_probs.values()))
    assert list(i_probs) == [test_env.NO_ACTION] + test_services


@pytest.mark.parametrize("attacker", list(ATTACKERS))
@pytest.mark.parametrize(
    "action,expected",
    [
        # no defender action, attacker compromises root node
        (0, ([1] * 6 + [1] + [0] * 8, 6, False)),
        # defender disables root node, attack surface empty, episode ends
        (1, ([0] + [1] * 5 + [0] * 9, 5, True)),
    ],
)
def test_env_first_step(attacker, action, expected, test_env_config):
    env = AttackSimulationEnv(dict(test_env_config, attacker_class=ATTACKERS[attacker]))
    env.seed(42)
    env.reset()
    obs, reward, done, _ = env.step(action)
    obs_, reward_, done_ = expected
    assert all(np.array(obs) == obs_) and (reward == reward_) and (done == done_)


def test_env_render(test_env):
    assert test_env.render() is None or test_env.render() is True


def test_env_empty_config():
    env = AttackSimulationEnv(dict())
    # defaults for project graph
    assert all(np.array(env.reset()) == [1] * 18 + [0] * 78)


@pytest.mark.parametrize("save_graphs", [False, True])
@pytest.mark.parametrize("save_text", [False, True])
def test_env_render_save_graphs(save_graphs, save_text, test_env_config, tmpdir):
    env = AttackSimulationEnv(dict(test_env_config, save_graphs=save_graphs, save_text=save_text))
    seed = 42
    with tmpdir.as_cwd():
        env.seed(seed)
        env.reset()
        env.done = True
        env.render()
        render_dir = tmpdir.join("render").join(f"{seed}_1")
        files = render_dir.listdir()

        base = "render"

        num_files = 2 * int(save_graphs) + int(save_text)

        assert len(files) == num_files

        basenames = [f.basename for f in files]

        condition = f"{base}_frames" in basenames and "render.html" in basenames
        if save_graphs:
            assert condition
        else:
            assert not condition

        condition = "log.txt" in basenames
        if save_text:
            assert condition
        else:
            assert not condition
