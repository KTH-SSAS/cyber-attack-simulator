from attack_simulator.config import AgentConfig, EnvironmentConfig
from attack_simulator.agents.policy_agents import ReinforceAgent
from attack_simulator.agents.baseline_agents import RuleBasedAgent
from attack_simulator.agents.baseline_agents import RandomMCAgent
from attack_simulator.attack_simulation_env import AttackSimulationEnv
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import logging
import numpy as np
import torch
import time
import random
from dataclasses import asdict

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
class Runner:

    def __init__(self, agent_config: AgentConfig, env_config: AttackSimulationEnv, include_services=False, random_seed=0, use_cuda=False):

        self.include_services = include_services #TODO move this to the environment
        self.use_cuda = use_cuda
        self.random_seed = random_seed
        self.agent_config = agent_config
        self.env = create_environment(env_config)
        self.agent = create_agent(agent_config, self.env, use_cuda)

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

            if self.include_services:
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

    def train(self, episodes, plot=True):
        start = time.time()
        num_compromised_flags = 0
        returns, losses, lengths, num_compromised_flags = self.run_multiple_episodes(episodes, plot=plot)
        duration = time.time() - start
        return duration, returns, losses, lengths, num_compromised_flags

    def evaluate(self, evaluation_rounds=50, plot=True):
        start = time.time()
        with torch.no_grad():
            returns, losses, lengths, num_compromised_flags = self.run_multiple_episodes(evaluation_rounds, evaluation=True)
        duration = time.time() - start
        return duration, returns, losses, lengths, num_compromised_flags
        

    def train_and_evaluate(self, episodes, evaluation_rounds=0, plot=True):
        log = logging.getLogger("trainer")
        training_duration, returns, losses, lengths, num_compromised_flags = self.train(episodes, plot=plot)
        duration = training_duration
        if evaluation_rounds > 0:
            evaluation_duration, returns, losses, lengths, num_compromised_flags = self.evaluate(evaluation_rounds=evaluation_rounds, plot=plot)
            duration += evaluation_duration
        log.debug(f"Total elapsed time: {duration}, agent time: {self.agent_time}, environment time: {self.environment_time}")
        return duration, returns, losses, lengths, num_compromised_flags

    def computational_complexity(self, start_episodes=100, end_episodes=5, step_episodes=-5):
        log = logging.getLogger("trainer")
        episodes_list = range(start_episodes, end_episodes, step_episodes)
        simulation_time_list = []
        for episodes in episodes_list:
            self.agent = create_agent(self.agent_config)
            simulation_time_list.append(self.train_and_evaluate(episodes, plot=False))
            
            log.debug(f"Simulation time {simulation_time_list} as a function of number of episodes {episodes_list}.")

        fig, ax = plt.subplots()
        title = "Computational complexity"
        ax.set_title(title)
        ax.plot(episodes_list, simulation_time_list, '.', color='black')
        # ax1.set_xlabel("Episode")
        ax.set_ylabel("Time")
        ax.set_xlabel("Episodes")
        fig.savefig('computational_complexity.pdf', dpi=200)
        plt.show()
        return (episodes_list, simulation_time_list)

    def effect_of_measurement_accuracy_on_returns(self, episodes=10000, evaluation_rounds=50, resolution=5):
        log = logging.getLogger("trainer")
        # Training on perfect obbservations
        duration, returns, losses, lengths, num_compromised_flags = self.train(episodes, plot=False)
        returns_matrix = np.zeros((resolution, resolution))
        fp_array = np.zeros((resolution, resolution))
        tp_array = np.zeros((resolution, resolution))
        for fp in range(0, resolution):
            for tp in range(0, resolution):
                random.seed(self.random_seed)
                torch.manual_seed(self.random_seed)
                self.env.attack_graph.false_positive = fp/(resolution-1)
                self.env.attack_graph.true_positive = tp/(resolution-1)
                fp_array[fp, tp] = self.env.attack_graph.false_positive
                tp_array[fp, tp] = self.env.attack_graph.true_positive
                self.env.attack_graph.reset()

                self.agent = create_agent(self.agent_config)
                duration, returns, losses, lengths, num_compromised_flags = self.evaluate(evaluation_rounds=evaluation_rounds, plot=False)

                returns_matrix[fp, tp] = sum(returns)/len(returns)
                log.debug(f"fp=\n{fp_array}, tp=\n{tp_array}, returns_matrix=\n{returns_matrix}")
                print(f"returns_matrix=\n{returns_matrix}")
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Plot the surface.
        surf = ax.plot_surface(fp_array, tp_array, returns_matrix, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        fig.savefig('3D.pdf', dpi=200)

        plt.show()

        return returns_matrix

    def generate_graphviz_file(self):
        self.env.attack_graph.generate_graphviz_file()
