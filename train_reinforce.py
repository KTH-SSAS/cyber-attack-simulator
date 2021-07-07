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
	parser.add_argument('-e', '--early_flag_reward', type=int, default=1000, help='Flag reward for the attacker when capturing flags early in the attack graph (use positive values).')
	parser.add_argument('-l', '--late_flag_reward', type=int, default=10000, help='Flag reward for the attacker when capturing flags late in the attack graph (use positive values).')
	parser.add_argument('-r', '--random_seed', type=int, default=0, help='Random seed for both numpy and torch.')
	parser.add_argument('-w', '--hidden_width', type=int, default=64, help='Dimension of the hidden linear layers.')
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

	env = AttackSimulationEnv(deterministic=args.deterministic, early_flag_reward=args.early_flag_reward, late_flag_reward=args.late_flag_reward, graph_size=args.graph_size)

	if args.graph:
		env.attack_graph.generate_graphviz_file()

	
	services = 17
	agent = ReinforceAgent(attack_steps, services, hidden_dim=args.hidden_width)

	run_multiple_simulations(args.n_simulations, env, agent)
	if args.test:
		test(env, agent, graph_size=args.graph_size)

