import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from .config import AgentConfig, EnvConfig
from .runner import Runner
from .utils import create_agent, create_env, set_seeds

logger = logging.getLogger("trainer")


class Analyzer:
    """Metaclass to manage different forms of runs"""

    def __init__(self, args) -> None:
        self.env_config = self.create_env_config(args)
        env = create_env(self.env_config)

        self.agent_config = self.create_agent_config(args, env.attack_graph)
        agent = create_agent(self.agent_config, env.attack_graph)

        self.runner = Runner(agent, env)

    def create_env_config(self, args):
        return EnvConfig(
            args.deterministic,
            args.early_flag_reward,
            args.late_flag_reward,
            args.final_flag_reward,
            args.easy_ttc,
            args.hard_ttc,
            args.graph_size,
            args.attacker_strategy,
            args.true_positive_training,
            args.false_positive_training,
        )

    def create_agent_config(self, args, graph):
        return AgentConfig(
            agent_type=args.defender,
            random_seed=args.random_seed,
            input_dim=graph.num_attacks + graph.num_services,
            hidden_dim=args.hidden_width,
            num_actions=graph.num_services + 1,
            learning_rate=args.lr,
            use_cuda=args.cuda,
            attack_graph=graph,
        )

    def train_and_evaluate(
        self,
        episodes,
        rollouts=0,
        tp_train=1.0,
        fp_train=0.0,
        tp_eval=1.0,
        fp_eval=0.0,
        plot=True,
    ):
        runner = self.runner
        duration = 0
        if runner.agent.trainable:  # Don't train untrainable agents
            runner.env.update_accuracy(tp_train, fp_train)
            training_duration, returns, losses, lengths, num_compromised_flags = runner.train(
                episodes, plot=plot
            )
            duration = training_duration
        if rollouts > 0:
            runner.env.update_accuracy(tp_eval, fp_eval)
            evaluation_duration, returns, losses, lengths, num_compromised_flags = runner.evaluate(
                rollouts, plot=False
            )
            duration += evaluation_duration
        logger.debug(
            f"Total elapsed time: {duration}, agent time: {runner.agent_time},"
            f" environment time: {runner.env_time}"
        )
        return duration, returns, losses, lengths, num_compromised_flags

    def clean_simulation(self, training_episodes, evaluation_episodes):
        # TODO Create a new runner, new agent and new environment
        # to make sure no state remains from previous simulations.
        # But I think  the provision of a clean simulation should be located in the Runner.
        env = create_env(self.env_config)
        agent = create_agent(self.agent_config, env.attack_graph)
        runner = Runner(agent, env)
        duration = 0
        if agent.trainable:  # Don't train untrainable agents
            duration, returns, losses, lengths, num_compromised_flags = runner.train(
                training_episodes, plot=False
            )
        evaluation_duration, returns, losses, lengths, num_compromised_flags = runner.evaluate(
            evaluation_episodes, plot=False
        )
        duration += evaluation_duration
        return duration, returns, losses, lengths, num_compromised_flags

    def simulations_with_different_seeds(
        self, seeds, training_episodes=10000, evaluation_episodes=100
    ):
        """Histogram over returns for different random seeds."""
        mean_returns = list()
        i = 0
        for seed in seeds:
            i += 1
            print(f"Simulation {i}/{len(seeds)}")
            # FIXME: seed is never used!!
            duration, returns, losses, lengths, num_compromised_flags = self.clean_simulation(
                training_episodes, evaluation_episodes
            )
            mean_returns.append(np.mean(returns))

        fig = plt.figure()
        n, bins, patches = plt.hist(mean_returns, 50, density=True, facecolor="g", alpha=0.75)

        plt.xlabel("Mean returns")
        plt.ylabel("Frequency")
        plt.title("Histogram mean returns for different seeds")
        plt.grid(True)
        fig.savefig("Histogram of random seeds.pdf", dpi=200)
        plt.show()

        return mean_returns

    def effect_of_hidden_layer_size_on_return(
        self, training_episodes=10000, evaluation_episodes=100
    ):
        """Plot the returns as a function of the size of the hidden layer and the graph size."""
        hidden_layer_sizes = [16, 64, 256]
        graph_sizes = ["large", "medium", "small"]
        n_attack_steps = [7, 29, 78]  # TODO These shouldn't be hard-coded here.

        # This function doesn't work with 0 evaluation episodes
        if evaluation_episodes != 0:
            eval_episodes = evaluation_episodes
        else:
            eval_episodes = 100

        shape = (len(hidden_layer_sizes), len(graph_sizes))
        hls_array = np.zeros(shape)
        gs_array = np.zeros(shape)
        returns_matrix = np.zeros(shape)
        for graph_size_index in range(len(graph_sizes)):
            graph_size = graph_sizes[graph_size_index]
            # Simulating with rule-based agent as baseline.
            self.agent_config.agent_type = "rule-based"
            # FIXME: this is independent of graph_size!!
            duration, returns, losses, lengths, num_compromised_flags = self.clean_simulation(
                2, eval_episodes
            )
            mean_rule_based_returns = sum(returns) / len(returns)
            for hidden_layer_size_index in range(0, len(hidden_layer_sizes)):
                hidden_layer_size = hidden_layer_sizes[hidden_layer_size_index]
                # Simulating with reinforcement agent
                self.env_config.graph_size = graph_size
                self.agent_config.agent_type = "reinforce"
                self.agent_config.hidden_dim = hidden_layer_size
                duration, returns, losses, lengths, num_compromised_flags = self.clean_simulation(
                    training_episodes, eval_episodes
                )
                mean_reinforce_returns = sum(returns) / len(returns)

                hls_array[graph_size_index, hidden_layer_size_index] = hidden_layer_size
                gs_array[graph_size_index, hidden_layer_size_index] = n_attack_steps[
                    graph_size_index
                ]
                returns_matrix[graph_size_index, hidden_layer_size_index] = (
                    mean_reinforce_returns / mean_rule_based_returns
                )
                logger.debug("returns_matrix")
                logger.debug(returns_matrix)

        fig = plt.figure()
        ax = fig.gca(projection="3d")

        # Plot the surface.
        ax.plot_surface(
            hls_array, gs_array, returns_matrix, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )

        ax.set_xlabel("Hidden layer size")
        ax.set_ylabel("Graph size")
        ax.set_zlabel("Returns")
        fig.savefig("Hidden layer and graph size", dpi=200)
        # fig.savefig('3D.jpg', dpi=200)

        plt.show()

    def effect_of_size_on_returns(
        self, training_episodes=10000, evaluation_episodes=100, random_seed_min=0, random_seed_max=1
    ):
        """
        Plot the returns as a function of graph size for different agents.
        Do this for multiple random seeds.
        """
        agent_types = ["reinforce", "rule-based", "random"]
        graph_sizes = ["small", "medium", "large"]
        n_attack_steps = [7, 29, 78] * (
            random_seed_max - random_seed_min
        )  # TODO These shouldn't be hard-coded here.
        colors = ["red", "blue", "green"] * (random_seed_max - random_seed_min)

        if evaluation_episodes != 0:
            eval_episodes = evaluation_episodes
        else:
            eval_episodes = 100

        mean_returns = dict()
        # The inertial agent performs so poorly that we ignore it,
        # because the graphs become illegible.
        # agent_types = ['reinforce', 'rule-based', 'random', 'inertial']
        for agent_type in agent_types:
            mean_returns[agent_type] = list()
            for random_seed in range(random_seed_min, random_seed_max):
                for graph_size in graph_sizes:
                    print(
                        f"agent_type = {agent_type}, random_seed = {random_seed},"
                        f" graph_size = {graph_size}"
                    )
                    set_seeds(random_seed)
                    self.env_config.graph_size = graph_size
                    env = create_env(self.env_config)
                    self.agent_config.agent_type = agent_type
                    agent = create_agent(self.agent_config, env.attack_graph)
                    runner = Runner(agent, env)
                    if agent_type == "reinforce":
                        duration, returns, losses, lengths, num_compromised_flags = runner.train(
                            training_episodes, plot=False
                        )
                    else:
                        duration = 0
                    (
                        evaluation_duration,
                        returns,
                        losses,
                        lengths,
                        num_compromised_flags,
                    ) = runner.evaluate(eval_episodes, plot=False)
                    duration += evaluation_duration
                    mean_returns[agent_type].append(sum(returns) / len(returns))

        fig, ax = plt.subplots()
        title = "Returns vs graph size"
        ax.set_title(title)
        for i in range(0, len(agent_types)):
            ax.plot(n_attack_steps, mean_returns[agent_types[i]], ".", color=colors[i])

        ax.set_ylabel("Mean returns")
        ax.set_xlabel("Graph size")
        fig.savefig(f"returns_vs_size_seed_{random_seed}.pdf", dpi=200)
        plt.show()

    def computational_complexity(self, episodes_list=None):
        simulation_time_list = []
        if episodes_list is None:
            episodes_list = range(100, 5, -5)
        for episodes in episodes_list:
            self.runner.agent = create_agent(self.agent_config)
            data = self.train_and_evaluate(episodes, plot=False)
            simulation_time_list.append(data)
            logger.debug(
                f"Simulation time {simulation_time_list} as a function of"
                f" the number of episodes {episodes_list}."
            )

        fig, ax = plt.subplots()
        title = "Computational complexity"
        ax.set_title(title)
        ax.plot(episodes_list, simulation_time_list, ".", color="black")
        # ax1.set_xlabel("Episode")
        ax.set_ylabel("Time")
        ax.set_xlabel("Episodes")
        fig.savefig("computational_complexity.pdf", dpi=200)
        plt.show()
        return (episodes_list, simulation_time_list)

    def effect_of_measurement_accuracy_on_returns(
        self,
        episodes=10000,
        rollouts=50,
        tp_low=0.0,
        tp_high=1.0,
        fp_low=0.0,
        fp_high=1.0,
        resolution=5,
        random_seed=0,
    ):
        """Plot the returns as a function of share of true and false positives"""
        # Training on perfect obbservations
        runner = self.runner
        duration, returns, losses, lengths, num_compromised_flags = runner.train(
            episodes, plot=False
        )
        tps = np.linspace(tp_low, tp_high, resolution)
        fps = np.linspace(fp_low, fp_high, resolution)

        def mean_returns(tp, fp):
            set_seeds(random_seed)
            runner.env.update_accuracy(tp, fp)
            duration, returns, losses, lengths, num_compromised_flags = runner.evaluate(
                episodes=rollouts, plot=False
            )
            return np.mean(returns)

        returns_matrix = np.array([[mean_returns(tp, fp) for tp in tps] for fp in fps])
        logger.debug(f"returns_matrix=\n{returns_matrix}")

        tp_array, fp_array = np.meshgrid(tps, fps)

        fig = plt.figure()
        ax = fig.gca(projection="3d")

        # Plot the surface.
        ax.plot_surface(
            fp_array, tp_array, returns_matrix, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )

        ax.set_xlabel("% false positives")
        ax.set_ylabel("% true positives")
        ax.set_zlabel("Returns")
        fig.savefig(f"Accuracy seed {random_seed}.pdf", dpi=200)
        # fig.savefig('3D.jpg', dpi=200)

        plt.show()

        return returns_matrix
