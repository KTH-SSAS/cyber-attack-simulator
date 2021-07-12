from attack_simulator.agents.policy_agents import ReinforceAgent
from attack_simulator.agents.baseline_agents import RuleBasedAgent
from attack_simulator.agents.baseline_agents import RandomMCAgent
from attack_simulator.attack_simulation_env import AttackSimulationEnv
import matplotlib.pyplot as plt
import logging
import numpy.random as random
import numpy as np
import torch


class Runner:

    def __init__(self, agent_type, deterministic,  early_flag_reward, late_flag_reward, final_flag_reward, easy_ttc, hard_ttc, graph_size, attacker_strategy, true_positive, false_positive, input_dim, services, hidden_dim, allow_skip, include_services_in_state=False):

        self.env = AttackSimulationEnv(deterministic=deterministic, early_flag_reward=early_flag_reward,
                                  late_flag_reward=late_flag_reward, final_flag_reward=final_flag_reward, easy_ttc=easy_ttc, hard_ttc=hard_ttc, graph_size=graph_size, attacker_strategy=attacker_strategy, true_positive=true_positive, false_positive=false_positive)

        if agent_type == 'reinforce':
            self.agent = ReinforceAgent(input_dim, services,
                                   hidden_dim=hidden_dim, allow_skip=allow_skip)
        elif agent_type == 'rule_based':
            self.agent = RuleBasedAgent(self.env)
        elif agent_type == 'random':
            self.agent = RandomMCAgent(services, allow_skip=allow_skip)

        self.include_services_in_state = include_services_in_state

    def run_sim(self, plot_results=False):
        services = {}  # Serves as a key for which services belong to which index
        done = False
        for service, i in enumerate(self.env.attack_graph.enabled_services):
            services[service] = i

        enabled_services = np.ones(len(services), dtype=np.int8)

        rewards = []
        num_services = []
        compromised_flags = []
        state = self.env._next_observation()  # Intial state
        while not done:

            if self.include_services_in_state:
                state = np.concatenate([state, enabled_services])

            action = self.agent.act(state)

            if self.agent.can_skip:
                if action > 0:
                    # Shift action by 1 since action==0 is treated as skip
                    enabled_services[action - 1] = 0
                else:
                    pass  # Skip action and don't disable a service
            else:
                enabled_services[action] = 0

            new_state, reward, done, info = self.env.step(enabled_services)
            rewards.append(reward)
            # count number of running services
            num_services.append(sum(enabled_services))
            compromised_flags.append(len(info['compromised_flags']))
            state = new_state

        if plot_results:
            _, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
            ax1.plot(rewards, "b")
            ax1.set_ylabel("Reward")
            ax2.plot(num_services, "r")
            ax2.set_ylabel("Number of services")
            ax3.plot(compromised_flags)
            ax3.set_ylabel("Compromised flags")
            ax3.set_xlabel("Step")
            plt.show()

        return rewards, info['time'], info['compromised_flags']

    def run_multiple_simulations(self, episodes, evaluation=False, plot=True):

        log = logging.getLogger("trainer")
        returns = np.zeros(episodes)
        losses = np.zeros(episodes)
        lengths = np.zeros(episodes)
        num_compromised_flags = np.zeros(episodes)
        max_patience = 50
        patience = max_patience
        prev_loss = 1E6

        if evaluation:
            self.agent.eval()
        else:
            self.agent.train()

        try:
            for i in range(episodes):
                rewards, episode_length, compromised_flags = self.run_sim()
                if evaluation:
                    loss = self.agent.calculate_loss(rewards).item()
                else:
                    loss = self.agent.update(rewards)
                losses[i] = loss
                returns[i] = sum(rewards)
                lengths[i] = episode_length
                num_compromised_flags[i] = len(compromised_flags)
                self.env.reset()
                log.debug(
                    f"Episode: {i+1}/{episodes}, Loss: {loss}, Return: {sum(rewards)}, Episode Length: {episode_length}")

                if (prev_loss - loss) < 0.01 and not evaluation:
                    patience -= 1
                else:
                    patience = (
                        patience+1) if patience < max_patience else max_patience
                if patience == 0:
                    log.debug("Stopping due to insignicant loss changes.")
                    break

                prev_loss = loss

        except KeyboardInterrupt:
            print("Stopping...")

        if evaluation:
            log.debug(f"Average returns: {sum(returns)/len(returns)}")

        if plot:
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

            ax4.plot(num_compromised_flags)
            ax4.set_ylabel("Compromised flags")

            ax4.set_xlabel("Episode")
            fig.savefig('plot.pdf', dpi=200)
            plt.show()

        return returns, losses, lengths, num_compromised_flags

    def train_and_evaluate(self, n_simulations, evaluation_rounds=0):

        # Train
        self.run_multiple_simulations(n_simulations)

        # Evaluate
        if evaluation_rounds > 0:
            self.run_multiple_simulations(evaluation_rounds, evaluation=True, include_services=include_services_in_state)

    def explore_parameter(self, episodes, evaluation_rounds):

        self.run_multiple_simulations(
            episodes, evaluation=True, plot=False)

    def generate_graphviz_file(self):
            self.env.attack_graph.generate_graphviz_file()
