from attack_simulator.policy_agents import ReinforceAgent
from attack_simulator.attack_simulation_env import AttackSimulationEnv
import matplotlib.pyplot as plt
import logging
import numpy.random as random
import numpy as np
import torch


def run_sim(env: AttackSimulationEnv, agent: ReinforceAgent, plot_results=False, include_services_in_state=False):
    services = {} # Serves as a key for which services belong to which index
    done = False
    for service, i in enumerate(env.attack_graph.enabled_services):
        services[service] = i
   
    enabled_services = np.ones(len(services), dtype=np.int8)

    rewards = []
    num_services = []
    compromised_steps = []
    state = env._next_observation()  # Intial state
    while not done:

        if include_services_in_state:
            state = np.concatenate([state, enabled_services])

        action = agent.act(state)

        if agent.can_skip:
            if action > 0:
                enabled_services[action - 1] = 0 # Shift action by 1 since action==0 is treated as skip
            else:
                pass # Skip action and don't disable a service
        else:
            enabled_services[action] = 0
                        
        new_state, reward, done, info = env.step(enabled_services)
        rewards.append(reward)
        # count number of running services
        num_services.append(sum(enabled_services))
        compromised_steps.append(len(info['compromised_steps']))
        state = new_state

    if plot_results:
        _, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        ax1.plot(rewards, "b")
        ax1.set_ylabel("Reward")
        ax2.plot(num_services, "r")
        ax2.set_ylabel("Number of services")
        ax3.plot(compromised_steps)
        ax3.set_ylabel("Compromised steps")
        ax3.set_xlabel("Step")
        plt.show()

    return rewards, info['time'], info['compromised_steps']


def run_multiple_simulations(episodes, env: AttackSimulationEnv, agent: ReinforceAgent, evaluation=False, include_services=False):

    log = logging.getLogger("trainer")
    returns = np.zeros(episodes)
    losses = np.zeros(episodes)
    lengths = np.zeros(episodes)
    num_compromised_steps = np.zeros(episodes)
    max_patience = 50
    patience = max_patience
    prev_loss = 1E6

    if evaluation:
        agent.eval()
    else:
        agent.train()

    try:
        for i in range(episodes):
            rewards, episode_length, compromised_steps = run_sim(env, agent, include_services_in_state=include_services)
            if evaluation:
                loss = agent.calculate_loss(rewards).item()
            else:
                loss = agent.update(rewards)
            losses[i] = loss
            returns[i] = sum(rewards)
            lengths[i] = episode_length
            num_compromised_steps[i] = len(compromised_steps)
            env.reset()
            log.debug(
                f"Episode: {i+1}/{episodes}, Loss: {loss}, Return: {sum(rewards)}, Episode Length: {episode_length}")

            if (prev_loss - loss) < 0.01 and not evaluation:
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
    title = "Training Results" if not evaluation else "Evaluation Results"
    ax1.set_title(title)
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
