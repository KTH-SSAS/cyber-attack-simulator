#!/usr/bin/env python3

import argparse
import logging

import attack_simulator.analysis as analysis
from attack_simulator.config import AgentConfig, EnvironmentConfig
from attack_simulator.utils import set_seeds


def initialize(args):
    if args.deterministic:
        set_seeds(args.random_seed)
    return analysis.Analyzer(args)


def create_env_config(args, services):
    return EnvironmentConfig(
        args.deterministic,
        args.early_flag_reward,
        args.late_flag_reward,
        args.final_flag_reward,
        args.easy_ttc,
        args.hard_ttc,
        args.graph_size,
        services,
        args.attacker_strategy,
        args.true_positive_training,
        args.false_positive_training,
        args.true_positive_evaluation,
        args.false_positive_evaluation,
    )


def create_agent_config(args, attack_steps, services):
    if args.include_services:
        input_dim = attack_steps + services
    else:
        input_dim = attack_steps
    return AgentConfig(
        agent_type=args.agent,
        hidden_dim=args.hidden_width,
        learning_rate=args.lr,
        input_dim=input_dim,
        num_actions=services,
        allow_skip=(not args.no_skipping),
    )


def main(args):

    analyzer = initialize(args)

    if args.action == "train_and_evaluate":
        analyzer.train_and_evaluate(
            args.episodes,
            args.evaluation_rounds,
            tp_train=args.true_positive_training,
            fp_train=args.false_positive_training,
            tp_evaluate=args.true_positive_evaluation,
            fp_evaluate=args.false_positive_evaluation,
        )
    if args.action == "computational_complexity":
        analyzer.computational_complexity(100, 1, -1)
    if args.action == "accuracy":
        analyzer.effect_of_measurement_accuracy_on_returns(
            episodes=args.episodes,
            evaluation_rounds=args.evaluation_rounds,
            tp_low=args.true_positive_low,
            tp_high=args.true_positive_high,
            fp_low=args.false_positive_low,
            fp_high=args.false_positive_high,
            resolution=args.accuracy_resolution,
            random_seed=args.random_seed,
        )
    if args.action == "size":
        analyzer.effect_of_size_on_returns(
            training_episodes=args.episodes,
            evaluation_episodes=args.evaluation_rounds,
            random_seed_min=0,
            random_seed_max=args.random_seed,
        )
    if args.action == "hidden":
        analyzer.effect_of_hidden_layer_size_on_return(
            training_episodes=args.episodes, evaluation_episodes=args.evaluation_rounds
        )
    if args.action == "seed":
        analyzer.simulations_with_different_seeds(
            [1] * 10, training_episodes=args.episodes, evaluation_episodes=args.evaluation_rounds
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reinforcement learning of a computer network defender."
    )
    parser.add_argument(
        "--action",
        choices=[
            "train_and_evaluate",
            "computational_complexity",
            "accuracy",
            "size",
            "hidden",
            "seed",
        ],
        type=str,
        default="train_and_evaluate",
        help="Select what action to perform.",
    )
    parser.add_argument("-g", "--graph", action="store_true", help="Generate a GraphViz .dot file.")
    parser.add_argument(
        "-d", "--deterministic", action="store_true", help="Make environment deterministic."
    )
    parser.add_argument(
        "-a",
        "--agent",
        choices=["reinforce", "rule_based", "inertial", "random"],
        type=str,
        default="reinforce",
        help='Select agent. Choices are "reinforce", "random", "inertial", and "rule_based".'
        '  "reinforce": a learning agent, "inertial": does nothing,'
        ' and "rule_based": near-maximixing based on white-box info about attack graph.',
    )
    parser.add_argument(
        "-t",
        "--attacker_strategy",
        choices=["value_maximizing", "random"],
        type=str,
        default="random",
        help='Select agent. Choices are "value_maximizing" and "random".',
    )
    parser.add_argument(
        "-s",
        "--graph_size",
        choices=["small", "medium", "large"],
        type=str,
        default="large",
        help='Run simulations on a "small", "medium" or "large" attack graph. Default is "large".',
    )
    parser.add_argument(
        "-n",
        "--episodes",
        type=int,
        default=10000,
        help="Maximum number of episodes. Default is 10000."
        " Training will stop automatically when losses are sufficiently low.",
    )
    parser.add_argument(
        "-e",
        "--early_flag_reward",
        type=int,
        default=10000,
        help="Flag reward for the attacker when capturing flags early in the attack graph."
        "Default is 10000. (Use positive values!)",
    )
    parser.add_argument(
        "-l",
        "--late_flag_reward",
        type=int,
        default=10000,
        help="Flag reward for the attacker when capturing flags late in the attack graph"
        "Default is 10000. (Use positive values!)",
    )
    parser.add_argument(
        "-f",
        "--final_flag_reward",
        type=int,
        default=10000,
        help="Flag reward for the attacker when capturing the final flag in the attack graph"
        "Default is 10000. (Use positive values!)",
    )
    parser.add_argument(
        "--easy_ttc",
        type=int,
        default=10,
        help="Mean time required by attacker to compromise easy attack steps. Default is 10.",
    )
    parser.add_argument(
        "--hard_ttc",
        type=int,
        default=100,
        help="Mean time required by attacker to compromise hard attack steps. Default is 100.",
    )
    parser.add_argument(
        "-r",
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for both numpy and torch. Default is 0.",
    )
    parser.add_argument(
        "-w",
        "--hidden_width",
        type=int,
        default=64,
        help="Dimension of the hidden linear layers. Defult is 64.",
    )
    parser.add_argument(
        "-m",
        "--evaluation_rounds",
        type=int,
        default=100,
        help="Number of simulations to run after training, for evaluation.",
    )
    parser.add_argument(
        "--true_positive_training",
        type=float,
        default=1.0,
        help="Probability that compromised attack steps are reported as compromised"
        " during training.",
    )
    parser.add_argument(
        "--false_positive_training",
        type=float,
        default=0.0,
        help="Probability that uncompromised attack steps are reported as compromised"
        "during training.",
    )
    parser.add_argument(
        "--true_positive_evaluation",
        type=float,
        default=1.0,
        help="Probability that compromised attack steps are reported as compromised"
        " during evaluation.",
    )
    parser.add_argument(
        "--false_positive_evaluation",
        type=float,
        default=0.0,
        help="Probability that uncompromised attack steps are reported as compromised"
        " during evaluation.",
    )
    parser.add_argument(
        "--false_positive_low",
        type=float,
        default=0.0,
        help="The lowest probability that uncompromised attack steps are reported as compromised."
        " Only valid when producing the accuracy plot (i.e. `--action accuracy`).",
    )
    parser.add_argument(
        "--false_positive_high",
        type=float,
        default=1.0,
        help="The highest probability that uncompromised attack steps are reported as compromised."
        " Only valid when producing the accuracy plot (i.e. `--action accuracy`).",
    )
    parser.add_argument(
        "--true_positive_low",
        type=float,
        default=0.0,
        help="The lowest probability that compromised attack steps are reported as compromised."
        " Only valid when producing the accuracy plot (i.e. `--action accuracy`).",
    )
    parser.add_argument(
        "--true_positive_high",
        type=float,
        default=1.0,
        help="The highest probability that compromised attack steps are reported as compromised."
        " Only valid when producing the accuracy plot (i.e. `--action accuracy`).",
    )
    parser.add_argument(
        "-c",
        "--accuracy_resolution",
        type=int,
        default=5,
        help="The number of data points for the accuracy graph (the same number for both axes).",
    )
    parser.add_argument(
        "--no_skipping", action="store_true", help="Do not add a skip action for the agent."
    )
    parser.add_argument(
        "--include_services", action="store_true", help="Include enabled services in the state."
    )
    parser.add_argument("--lr", help="Optimizer (Adam) learning rate.", default=1e-2)
    parser.add_argument("--cuda", action="store_true", help="Use CUDA acceleration.")
    arguments = parser.parse_args()

    logging.getLogger("simulator").setLevel(logging.DEBUG)
    logging.getLogger("simulator").addHandler(logging.FileHandler("simulator.log", mode="w"))
    logging.getLogger("trainer").setLevel(logging.DEBUG)
    logging.getLogger("trainer").addHandler(logging.FileHandler("trainer.log", mode="w"))

    main(arguments)

# vim: ft=python
