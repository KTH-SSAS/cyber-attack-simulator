from attack_simulator.utils import run_multiple_simulations
from attack_simulator.policy_agents import ReinforceAgent
from attack_simulator.attack_simulation_env import AttackSimulationEnv
from test.test_correctness import test
import logging
import numpy.random as random
import torch
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Reinforcement learning of a computer network defender.')
	parser.add_argument('-g', '--graph', action='store_true', help='Generate a GraphViz .dot file.')
	parser.add_argument('-d', '--deterministic', action='store_true', help='Make environment deterministic.')
	parser.add_argument('-t', '--test', action='store_true', help='Run tests.')
	parser.add_argument('-s', '--graph_size', choices=['small', 'medium', 'large'], type=str, default='large', help='Run simulations on a small, medium or large attack graph.')
	parser.add_argument('-n', '--n_simulations', type=int, default=10000, help='Number of simulations.')
	parser.add_argument('-f', '--flag_rewards', type=int, default=10000, help='Flag rewards for the attacker (use positive values).')
	parser.add_argument('-r', '--random_seed', type=int, default=0, help='Random seed for both numpy and torch.')
	parser.add_argument('-l', '--hidden_linear', type=int, default=64, help='Dimension of the hidden linear layers.')
	args = parser.parse_args()

	logging.getLogger("simulator").setLevel(logging.DEBUG)
	logging.getLogger("simulator").addHandler(logging.FileHandler("simulator.log", mode="w"))
	logging.getLogger("trainer").setLevel(logging.DEBUG)
	logging.getLogger("trainer").addHandler(logging.FileHandler("trainer.log", mode="w"))

	if args.graph_size == 'small':
		attack_steps = 7
	elif args.graph_size == 'medium':
		attack_steps = 29
	else:
		attack_steps = 78

	if args.deterministic:
		random.seed(args.random_seed)
		torch.manual_seed(args.random_seed)

	env = AttackSimulationEnv(deterministic=args.deterministic, flag_reward=args.flag_rewards, graph_size=args.graph_size)

	if args.graph:
		env.attack_graph.generate_graphviz_file()

	
	services = 17
	agent = ReinforceAgent(attack_steps, services, hidden_dim=args.hidden_linear)

	run_multiple_simulations(args.n_simulations, env, agent)
	if args.test:
		test(env, agent, graph_size=args.graph_size)

