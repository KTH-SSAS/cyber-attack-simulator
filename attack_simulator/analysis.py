from attack_simulator.config import AgentConfig
import logging
import matplotlib.pyplot as plt
from attack_simulator.runner import Runner
from attack_simulator.utils import set_seeds, create_agent
import numpy as np
from matplotlib import cm
import random


class Analyzer():
    """Metaclass to manage different forms of runs"""

    def __init__(self, runner, agent_config, use_cuda=False) -> None:
        self.runner: Runner = runner
        self.use_cuda = use_cuda
        # Save config to be able to reinitialize runner agent
        self.agent_config = agent_config

    def train_and_evaluate(self, episodes, evaluation_rounds=0, plot=True):
        log = logging.getLogger("trainer")
        runner = self.runner
        training_duration, returns, losses, lengths, num_compromised_flags = runner.train(
            episodes, plot=plot)
        duration = training_duration
        if evaluation_rounds > 0:
            evaluation_duration, returns, losses, lengths, num_compromised_flags = runner.evaluate(
                evaluation_rounds, plot=plot)
            duration += evaluation_duration
        log.debug(
            f"Total elapsed time: {duration}, agent time: {runner.agent_time}, environment time: {runner.environment_time}")
        return duration, returns, losses, lengths, num_compromised_flags

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
                    f"fp=\n{fp_array}, tp=\n{tp_array}, returns_matrix=\n{returns_matrix}")

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Plot the surface.
        surf = ax.plot_surface(fp_array, tp_array, returns_matrix, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        ax.set_xlabel("% false positives")
        ax.set_ylabel("% true positives")
        ax.set_zlabel("Returns")
        fig.savefig('3D.pdf', dpi=200)

        plt.show()

        return returns_matrix
