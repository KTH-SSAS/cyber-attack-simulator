#!/usr/bin/env python3

import argparse
import logging

from attack_simulator.agents import ATTACKERS, DEFENDERS
from attack_simulator.analysis import Analyzer
from attack_simulator.config import make_configs
from attack_simulator.graph import SIZES


def dict2choices(d):
    choices = list(d.keys())
    choices_help = '", "'.join(choices[:-1]) + f'" or "{choices[-1]}'
    return choices, choices_help


def parse_args():
    sizes, sizes_help = dict2choices(SIZES)
    defenders, defenders_help = dict2choices(DEFENDERS)
    attackers, attackers_help = dict2choices(ATTACKERS)

    parser = argparse.ArgumentParser(
        description="Reinforcement learning of a computer network defender."
    )

    parser.add_argument(
        "-a",
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
        "-D",
        "--defender",
        metavar="DEFENDER",
        choices=defenders,
        type=str,
        default=defenders[-1],
        help=f'Select defender. Choices are "{defenders_help}".  Default is "{defenders[-1]}"',
    )

    parser.add_argument(
        "-A",
        "--attacker",
        metavar="ATTACKER",
        choices=attackers,
        type=str,
        default=attackers[-1],
        help=f'Select attacker. Choices are "{attackers_help}".  Default is "{attackers[-1]}"',
    )

    parser.add_argument(
        "-s",
        "--graph_size",
        metavar="SIZE",
        choices=sizes,
        type=str,
        default=sizes[-1],
        help=f'Run simulations on a "{sizes_help}" attack graph. Default is "{sizes[-1]}".',
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
        "-l",
        "--low_flag_reward",
        type=int,
        default=1000,
        help="Flag reward for the attacker when capturing low-value flags in the attack graph."
        "Default is 1000. (Use positive values!)",
    )

    parser.add_argument(
        "-m",
        "--medium_flag_reward",
        type=int,
        default=10000,
        help="Flag reward for the attacker when capturing medium-value flags in the attack graph"
        "Default is 10000. (Use positive values!)",
    )

    # Override default help argument to free the `-h` short option
    parser.add_argument(
        "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit"
    )
    parser.add_argument(
        "-h",
        "--high_flag_reward",
        type=int,
        default=100000,
        help="Flag reward for the attacker when capturing high-value flags in the attack graph"
        "Default is 100000. (Use positive values!)",
    )

    parser.add_argument(
        "-E",
        "--easy_ttc",
        type=int,
        default=10,
        help="Mean time required by attacker to compromise easy attack steps. Default is 10.",
    )

    parser.add_argument(
        "-H",
        "--hard_ttc",
        type=int,
        default=100,
        help="Mean time required by attacker to compromise hard attack steps. Default is 100.",
    )

    parser.add_argument(
        "-r",
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for simulation. Default is None, which falls back to OS entropy.",
    )

    parser.add_argument(
        "-R",
        "--same_seed",
        action="store_true",
        help="Use the same seed for BOTH training AND evaluation. Defaults to `False`",
    )

    parser.add_argument(
        "-w",
        "--hidden_width",
        type=int,
        default=64,
        help="Dimension of the hidden linear layers (not used by all agents). Default is 64.",
    )

    parser.add_argument(
        "-N",
        "--rollouts",
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

    parser.add_argument("-L", "--lr", help="Optimizer (Adam) learning rate.", default=1e-2)

    parser.add_argument("-C", "--cuda", action="store_true", help="Use CUDA acceleration.")

    return parser.parse_args()


def main():
    parsed_args = parse_args()

    print(parsed_args)

    logging.getLogger("simulator").setLevel(logging.DEBUG)
    logging.getLogger("simulator").addHandler(logging.FileHandler("simulator.log", mode="w"))
    logging.getLogger("trainer").setLevel(logging.DEBUG)
    logging.getLogger("trainer").addHandler(logging.FileHandler("trainer.log", mode="w"))

    agent_config, env_config, graph_config = make_configs(parsed_args)

    print(agent_config)
    print(env_config)
    print(graph_config)

    analyzer = Analyzer(
        agent_config, env_config, graph_config, parsed_args.random_seed, parsed_args.same_seed
    )

    if parsed_args.action == "train_and_evaluate":
        analyzer.train_and_evaluate(
            parsed_args.episodes,
            parsed_args.rollouts,
            tp_train=parsed_args.true_positive_training,
            fp_train=parsed_args.false_positive_training,
            tp_eval=parsed_args.true_positive_evaluation,
            fp_eval=parsed_args.false_positive_evaluation,
        )

    if parsed_args.action == "computational_complexity":
        analyzer.computational_complexity(range(100, 1, -1))

    if parsed_args.action == "accuracy":
        analyzer.effect_of_measurement_accuracy_on_returns(
            episodes=parsed_args.episodes,
            rollouts=parsed_args.rollouts,
            tp_low=parsed_args.true_positive_low,
            tp_high=parsed_args.true_positive_high,
            fp_low=parsed_args.false_positive_low,
            fp_high=parsed_args.false_positive_high,
            resolution=parsed_args.accuracy_resolution,
        )

    if parsed_args.action == "size":
        analyzer.effect_of_size_on_returns(
            range(parsed_args.random_seed or 10),
            episodes=parsed_args.episodes,
            rollouts=parsed_args.rollouts,
        )

    if parsed_args.action == "hidden":
        analyzer.effect_of_hidden_layer_size_on_return(
            episodes=parsed_args.episodes,
            rollouts=parsed_args.rollouts,
        )

    if parsed_args.action == "seed":
        analyzer.simulations_with_different_seeds(
            range(parsed_args.random_seed or 10),
            episodes=parsed_args.episodes,
            rollouts=parsed_args.rollouts,
        )


if __name__ == "__main__":
    main()

# vim: ft=python
