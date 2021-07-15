from attack_simulator.config import AgentConfig, EnvironmentConfig
from attack_simulator.runner import Runner
import logging
import argparse
import numpy.random as random
import torch

def initialize(args):

    services = 18
    env_config = EnvironmentConfig(args.deterministic,
    args.early_flag_reward, args.late_flag_reward, args.final_flag_reward, args.easy_ttc, args.hard_ttc,
    args.graph_size, args.attacker_strategy, args.true_positive, args.false_positive)

    attack_steps = env_config.attack_steps

    if args.include_services:
        input_dim = attack_steps + services
    else:
        input_dim = attack_steps

    if args.deterministic:
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    
    agent_config = AgentConfig(agent_type=args.agent, hidden_dim=args.hidden_width,
    learning_rate=args.lr, input_dim=input_dim, num_actions=services, allow_skip=(not args.no_skipping))

    return Runner(agent_config, env_config, args.include_services, args.random_seed, args.cuda)

def main(args):

    runner = initialize(args)

    if args.action == 'train_and_evaluate':
        runner.train_and_evaluate(args.episodes, args.evaluation_rounds)
    if args.action == 'computational_complexity':
        runner.computational_complexity(100, 1, -1)
    if args.action == 'accuracy':
        runner.effect_of_measurement_accuracy_on_returns(episodes=args.episodes, evaluation_rounds=50, resolution=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Reinforcement learning of a computer network defender.')
    parser.add_argument('--action', choices=['train_and_evaluate', 'computational_complexity', 'accuracy'], type=str, default='train_and_evaluate',
                        help='Select what action to perform. Choices are "train_and_evaluate", "acccuracy" and "computational_complexity".')
    parser.add_argument('-g', '--graph', action='store_true',
                        help='Generate a GraphViz .dot file.')
    parser.add_argument('-d', '--deterministic', action='store_true',
                        help='Make environment deterministic.')
    parser.add_argument('-a', '--agent', choices=['reinforce', 'rule_based', 'random'], type=str, default='reinforce',
                        help='Select agent. Choices are "reinforce", "random" and "rule_based".')
    parser.add_argument('--attacker_strategy', choices=['value_maximizing', 'random'], type=str, default='random',
                        help='Select agent. Choices are "value_maximizing" and "random".')
    parser.add_argument('-s', '--graph_size', choices=['small', 'medium', 'large'], type=str, default='large',
                        help='Run simulations on a "small", "medium" or "large" attack graph. Default is "large".')
    parser.add_argument('-n', '--episodes', type=int, default=10000,
                        help='Maximum number of episodes. Training will stop automatically when losses are sufficiently low. Default is 10000.')
    parser.add_argument('-e', '--early_flag_reward', type=int, default=10000,
                        help='Flag reward for the attacker when capturing flags early in the attack graph (use positive values). Default is 10000.')
    parser.add_argument('-l', '--late_flag_reward', type=int, default=10000,
                        help='Flag reward for the attacker when capturing flags late in the attack graph (use positive values). Default is 10000.')
    parser.add_argument('-f', '--final_flag_reward', type=int, default=10000,
                        help='Flag reward for the attacker when capturing the final flag in the attack graph (use positive values). Default is 10000.')
    parser.add_argument('--easy_ttc', type=int, default=10,
                        help='Mean time required by attacker to compromise easy attack steps. Default is 10.')
    parser.add_argument('--hard_ttc', type=int, default=100,
                        help='Mean time required by attacker to compromise hard attack steps. Default is 100.')
    parser.add_argument('-r', '--random_seed', type=int, default=0,
                        help='Random seed for both numpy and torch. Default is 0.')
    parser.add_argument('-w', '--hidden_width', type=int, default=64,
                        help='Dimension of the hidden linear layers. Defult is 64.')
    parser.add_argument('--evaluation_rounds', type=int, default=0,
                        help='Number of simulations to run after training, for evaluation.')
    parser.add_argument('--true_positive', type=float, default=1.0,
                        help='Probability that compromised attack steps are reported as compromised.')
    parser.add_argument('--false_positive', type=float, default=0.0,
                        help='Probability that uncompromised attack steps are reported as compromised.')
    parser.add_argument('--no_skipping', action='store_true',
                        help="Do not add a skip action for the agent.")
    parser.add_argument('--include_services', action='store_true',
                        help="Include enabled services in the state.")
    parser.add_argument('--lr', help='Optimizer (Adam) learning rate.', default=1e-2)
    parser.add_argument('--cuda', action='store_true', help='Use CUDA acceleration.')
    arguments = parser.parse_args()

    logging.getLogger("simulator").setLevel(logging.DEBUG)
    logging.getLogger("simulator").addHandler(
        logging.FileHandler("simulator.log", mode="w"))
    logging.getLogger("trainer").setLevel(logging.DEBUG)
    logging.getLogger("trainer").addHandler(
        logging.FileHandler("trainer.log", mode="w"))

    main(arguments)
