import torch
import numpy.random
import random
from attack_simulator.config import AgentConfig, EnvironmentConfig
from attack_simulator.agents.policy_agents import ReinforceAgent
from attack_simulator.agents.baseline_agents import RuleBasedAgent
from attack_simulator.agents.baseline_agents import RandomMCAgent
from attack_simulator.attack_simulation_env import AttackSimulationEnv
from dataclasses import asdict
import matplotlib.pyplot as plt

def set_seeds(s):
    torch.manual_seed(s)
    numpy.random.seed(s)
    random.seed(s)

def create_environment(config: EnvironmentConfig):
    return AttackSimulationEnv(**asdict(config))

def create_agent(config: AgentConfig, env: AttackSimulationEnv = None, use_cuda=False):
    agent = None
    if config.agent_type == 'reinforce':
        agent = ReinforceAgent(config.input_dim, config.num_actions,
                                    config.hidden_dim, config.learning_rate, allow_skip=config.allow_skip, use_cuda=use_cuda)                                      
    elif config.agent_type == 'rule_based':
        agent = RuleBasedAgent(env)
    elif config.agent_type == 'random':
        agent = RandomMCAgent(config.num_actions, allow_skip=config.allow_skip)
    
    return agent

def plot_training_results(returns, losses, lengths, num_compromised_flags, evaluation, cutoff):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    title = "Training Results" if not evaluation else "Evaluation Results"
    ax1.set_title(title)
    ax1.plot(returns)
    # ax1.set_xlabel("Episode")
    ax1.set_xlim(0, cutoff)  # Cut off graph at stopping point
    ax1.set_ylabel("Return")
    ax2.plot(losses)
    ax2.set_ylabel('Loss')
    # ax2.set_xlabel('Episode')
    ax3.plot(lengths)
    ax3.set_ylabel("Episode Length")

    ax4.plot(num_compromised_flags)
    ax4.set_ylabel("Compromised flags")

    ax4.set_xlabel("Episode")
    fig.savefig('plot.pdf', dpi=200)
    plt.show()

def plot_episode(rewards, num_services, compromised_flags):
    _, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(rewards, "b")
    ax1.set_ylabel("Reward")
    ax2.plot(num_services, "r")
    ax2.set_ylabel("Number of services")
    ax3.plot(compromised_flags)
    ax3.set_ylabel("Compromised flags")
    ax3.set_xlabel("Step")
    plt.show()
