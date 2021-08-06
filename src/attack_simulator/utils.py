from dataclasses import asdict

import matplotlib.pyplot as plt

from .agents import DEFENDERS
from .config import AgentConfig, EnvConfig
from .env import AttackSimulationEnv
from .graph import AttackGraph


def create_env(config: EnvConfig):
    return AttackSimulationEnv(**asdict(config))


def create_agent(config: AgentConfig, graph: AttackGraph = None):
    config = asdict(config)
    if graph:
        input_dim = graph.num_attacks + graph.num_services
        num_actions = graph.num_services + 1
        config = dict(config, input_dim=input_dim, num_actions=num_actions, attack_graph=graph)
    print(config)
    return DEFENDERS[config["agent_type"]](config)


def plot_training_results(returns, losses, lengths, num_compromised_flags, evaluation, cutoff):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    title = "Training Results" if not evaluation else "Evaluation Results"
    ax1.set_title(title)
    ax1.plot(returns)
    # ax1.set_xlabel("Episode")
    ax1.set_xlim(0, cutoff)  # Cut off graph at stopping point
    ax1.set_ylabel("Return")
    ax2.plot(losses)
    ax2.set_ylabel("Loss")
    # ax2.set_xlabel('Episode')
    ax3.plot(lengths)
    ax3.set_ylabel("Episode Length")

    ax4.plot(num_compromised_flags)
    ax4.set_ylabel("Compromised flags")

    ax4.set_xlabel("Episode")
    fig.savefig("plot.pdf", dpi=200)
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
