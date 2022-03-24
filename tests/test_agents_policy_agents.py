import pytest

from attack_simulator.agents import ReinforceAgent
from attack_simulator.env import AttackSimulationEnv
from attack_simulator.rng import set_seeds


def _run_episodes(num_episodes, env: AttackSimulationEnv, agent, training=False):
    assert agent.trainable

    if hasattr(agent, "train"):
        agent.train(training)

    set_seeds(42)

    total_length = 0
    total_mean = 0.0
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_length = 0
        episode_mean = 0.0

        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            agent.update(obs, reward, done)
            episode_length += 1
            episode_mean += (reward - episode_mean) / episode_length

        if hasattr(agent, "loss"):
            assert agent.loss is not None

        total_length += episode_length
        total_mean += episode_length * (episode_mean - total_mean) / total_length

    return total_length, total_mean


def test_agents_policy_reinforce(env):
    agent = ReinforceAgent(
        dict(
            input_dim=env.observation_space.shape[0],
            hidden_dim=8,
            num_actions=env.action_space.n,
            learning_rate=1e-2,
            random_seed=42,
        )
    )

    num_episodes = 500

    # train
    _, train_mean = _run_episodes(num_episodes, env, agent, True)

    # evaluate
    _, eval_mean = _run_episodes(num_episodes, env, agent)

    # did we learn anything?
    assert train_mean <= eval_mean


# RuntimeError if torch was compiled with CUDA, but the host doesn't have it
# AssertionError when torch was compiled **without** CUDA
@pytest.mark.xfail(raises=(RuntimeError, AssertionError))
def test_agents_policy_reinforce_cuda(env):
    ReinforceAgent(
        dict(
            input_dim=env.observation_space.shape[0],
            hidden_dim=16,
            num_actions=env.action_space.n,
            learning_rate=1e-2,
            random_seed=42,
            use_cuda=True,
        )
    )
