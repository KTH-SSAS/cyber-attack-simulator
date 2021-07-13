from attack_simulator.agents.policy_agents import ReinforceAgent
from attack_simulator.agents.baseline_agents import RuleBasedAgent
from attack_simulator.agents.baseline_agents import RandomMCAgent
from attack_simulator.attack_simulation_env import AttackSimulationEnv
import matplotlib.pyplot as plt
import logging
import numpy.random as random
import numpy as np
import torch
import time

class Runner:

    def __init__(self, agent_type, deterministic, random_seed, early_flag_reward, late_flag_reward, final_flag_reward, easy_ttc, hard_ttc, graph_size, attacker_strategy, true_positive, false_positive, hidden_dim, learning_rate, no_skipping, include_services_in_state=False, use_cuda=False):

        if graph_size == 'small': 
            attack_steps = 7 
        elif graph_size == 'medium': 
            attack_steps = 29 
        elif graph_size == 'large': 
            attack_steps = 78 

        services = 18 

        if include_services_in_state: 
            input_dim = attack_steps + services
        else:
            input_dim = attack_steps

        if deterministic:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        allow_skip = not no_skipping

        self.env = AttackSimulationEnv(deterministic=deterministic, early_flag_reward=early_flag_reward,
                                       late_flag_reward=late_flag_reward, final_flag_reward=final_flag_reward, easy_ttc=easy_ttc, hard_ttc=hard_ttc, graph_size=graph_size, attacker_strategy=attacker_strategy, true_positive=true_positive, false_positive=false_positive)

        if agent_type == 'reinforce':
            self.agent = ReinforceAgent(input_dim, services,
                                        hidden_dim, learning_rate, allow_skip=allow_skip, use_cuda=use_cuda)                                      
        elif agent_type == 'rule_based':
            self.agent = RuleBasedAgent(self.env)
        elif agent_type == 'random':
            self.agent = RandomMCAgent(services, allow_skip=allow_skip)

        self.include_services_in_state = include_services_in_state

        self.agent_time = 0
        self.environment_time = 0

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

            agent_start = time.time()
            action = self.agent.act(state)
            self.agent_time += time.time() - agent_start

            if self.agent.can_skip:
                if action > 0:
                    # Shift action by 1 since action==0 is treated as skip
                    enabled_services[action - 1] = 0
                else:
                    pass  # Skip action and don't disable a service
            else:
                enabled_services[action] = 0

            env_start = time.time()
            new_state, reward, done, info = self.env.step(enabled_services)
            self.environment_time += time.time() - env_start

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

    def run_multiple_episodes(self, episodes, evaluation=False, plot=True):

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

    def train_and_evaluate(self, episodes, evaluation_rounds=0):

        log = logging.getLogger("trainer")
        start = time.time()

        # Train
        self.run_multiple_episodes(episodes)

        # Evaluate
        if evaluation_rounds > 0:
            self.run_multiple_episodes(
                evaluation_rounds, evaluation=True, include_services=self.include_services_in_state)

        duration = time.time() - start
        log.debug(f"Total elapsed time: {duration}, agent time: {self.agent_time}, environment time: {self.environment_time}")



    def effect_of_measurement_accuracy_on_returns(self, episodes=10000, evaluation_rounds=50, resolution=5):
        returns_matrix = np.zeros((resolution, resolution))
        for fp in range(0, resolution):
            for tp in range(0, resolution):
                self.env.attack_graph.false_positive = fp/(resolution-1)
                self.env.attack_graph.true_positive = tp/(resolution-1)
                returns, losses, lengths, num_compromised_flags = self.train_and_evaluate(episodes, evaluation_rounds)
                returns_matrix[fp, tp] = sum(returns)/len(returns)
                print(returns_matrix)
        return returns_matrix

    def generate_graphviz_file(self):
        self.env.attack_graph.generate_graphviz_file()
