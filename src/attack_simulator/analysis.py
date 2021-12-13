import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from attack_simulator.env import AttackSimulationEnv

from .config import AgentConfig, EnvConfig, GraphConfig, create_agent
from .graph import SIZES, AttackGraph
from .rng import set_seeds
from .runner import Runner

logger = logging.getLogger("trainer")


class Analyzer:
    """Metaclass to manage different forms of runs"""

    def __init__(
        self,
        agent_config: AgentConfig,
        env_config: EnvConfig,
        graph_config: GraphConfig,
        random_seed=None,
        same_seed=False,
    ):
        self.agent_config = agent_config
        self.env_config = env_config
        self.graph_config = graph_config

        self.seed_train, self.seed_eval = set_seeds(random_seed)
        self.same_seed = same_seed
        if same_seed:
            self.seed_eval = self.seed_train

    def train_and_evaluate(
        self,
        episodes,
        rollouts=0,
        fn_train=1.0,
        fp_train=0.0,
        fn_eval=1.0,
        fp_eval=0.0,
        plot=True,
    ):
        agent = create_agent(self.agent_config)

        timing = np.zeros(3)
        if agent.trainable:  # Don't train untrainable agents
            self.env_config.false_negative = fn_train
            self.env_config.false_positive = fp_train
            env = AttackSimulationEnv(self.env_config)
            results = Runner(agent, env).train(episodes, self.seed_train, plot=plot)
            # duration, agent_time, env_time
            timing = np.array(results[:3])

        if rollouts > 0:
            env = AttackSimulationEnv(
                self.env_config, false_negative=fn_eval, false_positive=fp_eval
            )
            results = Runner(agent, env).evaluate(rollouts, self.seed_eval, plot=False)
            # duration, agent_time, env_time
            timing += np.array(results[:3])

        logger.debug(
            f"Total elapsed time: {timing[0]}, agent: {timing[1]}, environment: {timing[2]}"
        )
        return timing[0]

    def simulations_with_different_seeds(self, seeds, episodes=10000, rollouts=100):
        """Histogram over returns for different random seeds."""
        env = AttackSimulationEnv(self.env_config)
        mean_returns = list()
        for i, seed in enumerate(seeds):
            print(f"Simulation {i+1}/{len(seeds)}")
            agent = create_agent(self.agent_config, random_seed=seed)
            runner = Runner(agent, env)

            if agent.trainable:  # Don't train untrainable agents
                runner.train(episodes, seed, plot=False)

            eval_seed = seed
            if not self.same_seed:
                eval_seed += self.seed_eval

            results = runner.evaluate(rollouts, eval_seed, plot=False)
            mean_returns.append(np.mean(results.returns))

        fig = plt.figure()
        n, bins, patches = plt.hist(mean_returns, 50, density=True, facecolor="g", alpha=0.75)

        plt.xlabel("Mean returns")
        plt.ylabel("Frequency")
        plt.title("Histogram mean returns for different seeds")
        plt.grid(True)
        fig.savefig("Histogram of random seeds.pdf", dpi=200)
        plt.show()

        return mean_returns

    def effect_of_hidden_layer_size_on_return(self, episodes=10000, rollouts=100):
        """Plot the returns as a function of the size of the hidden layer and the graph size."""
        hidden_layer_sizes = [16, 64, 256]
        graph_sizes = list(SIZES)

        # This function doesn't work with 0 evaluation episodes
        if rollouts == 0:
            rollouts = 100

        shape = (len(hidden_layer_sizes), len(graph_sizes))
        hls_array = np.zeros(shape)
        gs_array = np.zeros(shape)
        returns_matrix = np.zeros(shape)

        for graph_index, graph_size in enumerate(graph_sizes):
            self.env_config.graph_config.graph_size = graph_size
            env = AttackSimulationEnv(self.env_config)

            # Use rule-based agent as baseline.  No training needed.
            agent = create_agent(
                self.agent_config,
                agent_type="rule-based",
                attack_graph=env.g,
                input_dim=len(env.observation_space.spaces),
                num_actions=env.action_space.n,
            )

            results = Runner(agent, env).evaluate(rollouts, self.seed_eval, plot=False)
            mean_rule_based_returns = np.mean(results.returns)

            for hidden_layer_index, hidden_layer_size in enumerate(hidden_layer_sizes):
                # Simulating with reinforcement agent.  Train and evaluate.
                agent = create_agent(
                    self.agent_config,
                    agent_type="reinforce",
                    attack_graph=env.g,
                    input_dim=len(env.observation_space.spaces),
                    hidden_dim=hidden_layer_size,
                    num_actions=env.action_space.n,
                )
                runner = Runner(agent, env)
                runner.train(episodes, self.seed_train, plot=False)
                results = runner.evaluate(rollouts, self.seed_eval, plot=False)

                mean_reinforce_returns = np.mean(results.returns)

                hls_array[graph_index, hidden_layer_index] = hidden_layer_size
                gs_array[graph_index, hidden_layer_index] = env.g.num_attacks
                returns_matrix[graph_index, hidden_layer_index] = (
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

    def effect_of_size_on_returns(self, seeds, episodes=10000, rollouts=100):
        """
        Plot the returns as a function of graph size for different agents.
        Do this for multiple random seeds.
        """
        graph_sizes = list(SIZES)
        agent_types = ["reinforce", "rule-based", "random"]
        colors = ["red", "blue", "green"]
        shapes = [".", "o", "x"]

        if rollouts == 0:
            rollouts = 100

        mean_returns = defaultdict(list)
        gs_list = []

        for graph_size in graph_sizes:
            self.graph_config.graph_size = graph_size
            graph = AttackGraph(self.graph_config)
            env = AttackSimulationEnv(self.env_config, attack_graph=graph)
            gs_list += [graph.num_attacks] * len(seeds)

            for agent_type in agent_types:
                for seed in seeds:
                    print(
                        f"graph_size = {graph_size}, "
                        f"agent_type = {agent_type}, random_seed = {seed}"
                    )

                    agent = create_agent(
                        self.agent_config,
                        agent_type=agent_type,
                        attack_graph=graph,
                        input_dim=len(env.observation_space.spaces),
                        num_actions=env.action_space.n,
                        random_seed=seed,
                    )
                    runner = Runner(agent, env)

                    if agent.trainable:
                        runner.train(episodes, seed, plot=False)

                    eval_seed = seed
                    if not self.same_seed:
                        eval_seed += self.seed_eval

                    results = runner.evaluate(rollouts, eval_seed, plot=False)
                    mean_returns[agent_type].append(np.mean(results.returns))

        fig, ax = plt.subplots()
        title = "Returns vs graph size"
        ax.set_title(title)
        for i, agent_type in enumerate(agent_types):
            ax.plot(gs_list, mean_returns[agent_type], shapes[i], color=colors[i])

        ax.set_ylabel("Mean returns")
        ax.set_xlabel("Graph size")
        fig.savefig(f"returns_vs_size_seed_{seeds[-1]}.pdf", dpi=200)
        plt.show()

    def computational_complexity(self, episodes_list=None):
        simulation_time_list = []
        if episodes_list is None:
            episodes_list = range(100, 5, -5)
        for episodes in episodes_list:
            duration = self.train_and_evaluate(episodes, plot=False)
            simulation_time_list.append(duration)
            logger.debug(
                f"Simulation time {simulation_time_list}"
                f" as a function of number of episodes {episodes_list}."
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
        fn_low=0.0,
        fn_high=1.0,
        fp_low=0.0,
        fp_high=1.0,
        resolution=5,
    ):
        """Plot the returns as a function of share of true and false positives"""
        # Training on perfect obbservations
        env = AttackSimulationEnv(self.env_config)
        agent = create_agent(self.agent_config)
        Runner(agent, env).train(episodes, self.seed_train, plot=False)

        fns = np.linspace(fn_low, fn_high, resolution)
        fps = np.linspace(fp_low, fp_high, resolution)

        def mean_returns(fn, fp):
            # Evaluate on a range of different observation qualities.
            self.env_config.false_positive = fp
            self.env_config.false_negative = fn
            env = AttackSimulationEnv(self.env_config)
            results = Runner(agent, env).evaluate(rollouts, self.seed_eval, plot=False)
            return np.mean(results.returns)

        returns_matrix = np.array([[mean_returns(fn, fp) for fn in fns] for fp in fps])
        logger.debug(f"returns_matrix=\n{returns_matrix}")

        fn_array, fp_array = np.meshgrid(fns, fps)

        fig = plt.figure()
        ax = fig.gca(projection="3d")

        # Plot the surface.
        ax.plot_surface(
            fp_array, fn_array, returns_matrix, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )

        ax.set_xlabel("% false positives")
        ax.set_ylabel("% true positives")
        ax.set_zlabel("Returns")
        fig.savefig(f"Accuracy seed {self.seed_train}.pdf", dpi=200)
        # fig.savefig('3D.jpg', dpi=200)

        plt.show()

        # FIXME: why return something that's not ever used?
        return returns_matrix
