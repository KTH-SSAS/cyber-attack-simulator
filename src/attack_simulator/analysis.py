import logging
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from .config import AgentConfig, EnvironmentConfig
from .runner import Runner
from .utils import create_agent, create_environment, set_seeds


class Analyzer():
    """Metaclass to manage different forms of runs"""

    def __init__(self, args) -> None:
        self.env_config = self.create_env_config(args)
        self.agent_config = self.create_agent_config(args, self.env_config.attack_steps, self.env_config.services)
        
        env = create_environment(self.env_config)
        agent = create_agent(self.agent_config, env=env, use_cuda=args.cuda)
        self.runner = Runner(agent, env, args.include_services)

        # TODO: Refactor this. Save config to be able to reinitialize runner agent
        self.use_cuda = args.cuda
        self.agent_config = self.agent_config

    def create_env_config(self, args):
        return EnvironmentConfig(args.deterministic,
        args.early_flag_reward, args.late_flag_reward, args.final_flag_reward, args.easy_ttc, args.hard_ttc,
        args.graph_size, args.attacker_strategy, args.true_positive_training, args.false_positive_training, args.true_positive_evaluation, args.false_positive_evaluation)

    def create_agent_config(self, args, attack_steps, services):
        if args.include_services:
            input_dim = attack_steps + services
        else:
            input_dim = attack_steps
        return AgentConfig(agent_type=args.agent, hidden_dim=args.hidden_width,
        learning_rate=args.lr, input_dim=input_dim, num_actions=services, allow_skip=(not args.no_skipping))

    def train_and_evaluate(self, episodes, evaluation_rounds=0, tp_train=1.0, fp_train=0.0, tp_evaluate=1.0, fp_evaluate=0.0, plot=True):
        log = logging.getLogger("trainer")
        runner = self.runner
        runner.env.attack_graph.false_positive = fp_train
        runner.env.attack_graph.true_positive = tp_train
        training_duration, returns, losses, lengths, num_compromised_flags = runner.train(
            episodes, plot=plot)
        duration = training_duration
        if evaluation_rounds > 0:
            runner.env.update_accuracy(tp_evaluate, fp_evaluate)
            evaluation_duration, returns, losses, lengths, num_compromised_flags = runner.evaluate(
                evaluation_rounds, plot=False)
            duration += evaluation_duration
        log.debug(
            f"Total elapsed time: {duration}, agent time: {runner.agent_time}, environment time: {runner.environment_time}")
        return duration, returns, losses, lengths, num_compromised_flags

    def effect_of_size_on_returns(self, training_episodes=10000, evaluation_episodes=100, random_seed_min=0, random_seed_max=1):
        log = logging.getLogger("trainer")

        n_attack_steps = [7, 29, 78]*(random_seed_max-random_seed_min)
        colors = ['black', 'blue', 'red']*(random_seed_max-random_seed_min)
        agent_types = ['reinforce', 'rule_based', 'random']
        graph_sizes = ['small', 'medium', 'large'] 
    
        if evaluation_episodes != 0:
            eval_epidodes = evaluation_episodes
        else:
            eval_episodes = 100

        mean_returns = dict()
        # The inertial agent performs so poorly that we ignore it, because the graphs become illegible.
        #agent_types = ['reinforce', 'rule_based', 'random', 'inertial']
        for agent_type in agent_types:
            mean_returns[agent_type] = list()
            for random_seed in range(random_seed_min,random_seed_max):
                for graph_size in graph_sizes:
                    print(f"random_seed = {random_seed}, agent_type = {agent_type}, graph_size = {graph_size}")
                    set_seeds(random_seed)
                    self.env_config.graph_size = graph_size
                    env = create_environment(self.env_config)
                    self.agent_config.agent_type = agent_type
                    self.agent_config.input_dim = self.env_config.attack_steps
                    agent = create_agent(self.agent_config, env=env)
                    runner = Runner(agent, env) 
                    duration, returns, losses, lengths, num_compromised_flags = runner.train(
                        training_episodes, plot=False)
                    evaluation_duration, returns, losses, lengths, num_compromised_flags = runner.evaluate(
                        eval_episodes, plot=False)
                    duration += evaluation_duration
                    print(returns)
                    mean_returns[agent_type].append(sum(returns)/len(returns))

        fig, ax = plt.subplots()
        title = "Returns vs graph size"
        ax.set_title(title)
        print(n_attack_steps)
        print(mean_returns)
        print(agent_types)
        print(colors)
        for i in range(0,len(agent_types)):
            ax.plot(n_attack_steps, mean_returns[agent_types[i]], '.', color=colors[i])
        
        ax.set_ylabel("Mean returns")
        ax.set_xlabel("Graph size")
        fig.savefig(f'returns_vs_size_seed_{random_seed}.pdf', dpi=200)
        plt.show()

    def computational_complexity(self, start_episodes=100, end_episodes=5, step_episodes=-5):
        log = logging.getLogger("trainer")
        episodes_list = range(start_episodes, end_episodes, step_episodes)
        simulation_time_list = []
        for episodes in episodes_list:
            self.runner.agent = create_agent(self.agent_config, self.use_cuda)
            data = self.train_and_evaluate(episodes, plot=False)
            simulation_time_list.append(data)
            log.debug(
                f"Simulation time {simulation_time_list} as a function of number of episodes {episodes_list}.")

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

    def effect_of_measurement_accuracy_on_returns(self, episodes=10000, evaluation_rounds=50, tp_low=0.0, tp_high=1.0, fp_low=0.0, fp_high=1.0, resolution=5, random_seed=0):
        log = logging.getLogger("trainer")
        # Training on perfect obbservations
        runner = self.runner
        duration, returns, losses, lengths, num_compromised_flags = runner.train(
            episodes, plot=False)
        returns_matrix = np.zeros((resolution, resolution))
        fp_array = np.zeros((resolution, resolution))
        tp_array = np.zeros((resolution, resolution))
        for fp_index in range(0, resolution):
            for tp_index in range(0, resolution):
                set_seeds(random_seed)
                runner.env.attack_graph.false_positive = fp_low + \
                    (fp_high - fp_low)*fp_index/(resolution-1)
                runner.env.attack_graph.true_positive = tp_low + \
                    (tp_high - tp_low)*tp_index/(resolution-1)
                fp_array[fp_index,
                         tp_index] = runner.env.attack_graph.false_positive
                tp_array[fp_index,
                         tp_index] = runner.env.attack_graph.true_positive
                runner.env.attack_graph.reset()
                # Evaluate on a range of different observation qualities.
                duration, returns, losses, lengths, num_compromised_flags = runner.evaluate(
                    episodes=evaluation_rounds, plot=False)
                returns_matrix[fp_index, tp_index] = np.mean(returns)
                log.debug(
                    f"returns_matrix=\n{returns_matrix}")

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Plot the surface.
        surf = ax.plot_surface(fp_array, tp_array, returns_matrix, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        ax.set_xlabel("% false positives")
        ax.set_ylabel("% true positives")
        ax.set_zlabel("Returns")
        fig.savefig(f'Accuracy seed {random_seed}.pdf', dpi=200)
        #fig.savefig('3D.jpg', dpi=200)

        plt.show()

        return returns_matrix
