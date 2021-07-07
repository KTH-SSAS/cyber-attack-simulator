from attack_simulator.policy_agents import ReinforceAgent
from attack_simulator.attack_simulation_env import AttackSimulationEnv
import matplotlib.pyplot as plt
import logging
import numpy.random as random
import numpy as np
import torch


def run_sim(env: AttackSimulationEnv, agent: ReinforceAgent, plot_results=False):
    enabled_services: dict = {}
    done = False
    for service in env.attack_graph.enabled_services:
        enabled_services[service] = 1

    rewards = []
    num_services = []
    state = env._next_observation()  # Intial state
    while not done:
        action = agent.act(state)
        if action > 0:
                for i, key in enumerate(enabled_services):
                    if i == action - 1:
                        enabled_services[key] = 0
                        break
        new_state, reward, done, info = env.step(
            tuple(enabled_services.values()))
        rewards.append(reward)
        # count number of running services
        num_services.append(sum(list(enabled_services.values())))
        state = new_state

    if plot_results:
        _, ax = plt.subplots()
        twin1 = ax.twinx()
        ax.plot(rewards, "b")
        twin1.plot(num_services, "r")
        plt.show()

    return rewards, info['time'], info['compromised_steps']


def run_multiple_simulations(episodes, env: AttackSimulationEnv, agent: ReinforceAgent):

    log = logging.getLogger("trainer")
    returns = np.zeros(episodes)
    losses = np.zeros(episodes)
    lengths = np.zeros(episodes)
    num_compromised_steps = np.zeros(episodes)
    max_patience = 50
    patience = max_patience
    prev_loss = 1E6
    try:
        for i in range(episodes):
            rewards, episode_length, compromised_steps = run_sim(env, agent)
            loss = agent.update(rewards)
            losses[i] = loss
            returns[i] = sum(rewards)
            lengths[i] = episode_length
            num_compromised_steps[i] = len(compromised_steps)
            env.reset()
            log.debug(
                f"Episode: {i+1}/{episodes}, Loss: {loss}, Return: {sum(rewards)}, Episode Length: {episode_length}")

            if (prev_loss - loss) < 0.01:
                patience -= 1
            else:
                patience = (patience+1) if patience < max_patience else max_patience
            
            if patience == 0:
                log.debug("Stopping due to insignicant loss changes.")
                break
            
            prev_loss = loss

    except KeyboardInterrupt:
        print("Stopping...")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    ax1.plot(returns)
    # ax1.set_xlabel("Episode")
    ax1.set_xlim(0, i)  # Cut off graph at stopping point
    ax1.set_ylabel("Return")
    ax2.plot(losses)
    ax2.set_ylabel('Loss')
    # ax2.set_xlabel('Episode')
    ax3.plot(lengths)
    ax3.set_ylabel("Episode Length")

    ax4.plot(num_compromised_steps)
    ax4.set_ylabel("Compromised steps")

    ax4.set_xlabel("Episode")
    fig.savefig('plot.pdf', dpi=200)
    plt.show()
