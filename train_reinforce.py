from attack_simulator.utils import run_multiple_simulations
from attack_simulator.policy_agents import ReinforceAgent
from attack_simulator.attack_simulation_env import AttackSimulationEnv
from test.test_correctness import test_correctness
import logging
import numpy.random as random
import torch
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Reinforcement learning of a computer network defender.')
	parser.add_argument('-g', '--graph', action='store_true', help='Generate a GraphViz .dot file.')
	parser.add_argument('-d', '--deterministic', action='store_true', help='Make environment deterministic.')
	parser.add_argument('-t', '--test', action='store_true', help='Run tests.')
	parser.add_argument('-s', '--graph_size', choices=['small', 'medium', 'large'], type=str, default='large', help='Run simulations on a small, medium or large attack graph. Default is large.')
	parser.add_argument('-n', '--n_simulations', type=int, default=10000, help='Maximum number of simulations. Training will stop automatically when losses are sufficiently low. Default is 10000.')
	parser.add_argument('-e', '--early_flag_reward', type=int, default=1000, help='Flag reward for the attacker when capturing flags early in the attack graph (use positive values). Default is 1000.')
	parser.add_argument('-l', '--late_flag_reward', type=int, default=10000, help='Flag reward for the attacker when capturing flags late in the attack graph (use positive values). Default is 10000.')
	parser.add_argument('-f', '--final_flag_reward', type=int, default=100000, help='Flag reward for the attacker when capturing the final flag in the attack graph (use positive values). Default is 100000.')
	parser.add_argument('-r', '--random_seed', type=int, default=0, help='Random seed for both numpy and torch. Default is 0.')
	parser.add_argument('-w', '--hidden_width', type=int, default=64, help='Dimension of the hidden linear layers. Defult is 64.')
	parser.add_argument('--evaluation_rounds', type=int, default=0, help='Number of simulations to run after training, for evaluation.')
	parser.add_argument('--no_skipping', action='store_true', help="Do not add a skip action for the agent.")
	parser.add_argument('--include_services', action='store_true', help="Include enabled services in the state.")
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

	env = AttackSimulationEnv(deterministic=args.deterministic, early_flag_reward=args.early_flag_reward, late_flag_reward=args.late_flag_reward, final_flag_reward=args.final_flag_reward, graph_size=args.graph_size)

	if args.graph:
		env.attack_graph.generate_graphviz_file()

	
	services = 16
	allow_skip = not args.no_skipping # allowing skipping will add an additional 'skip' action
	include_services_in_state = args.include_services
	
	if include_services_in_state:
		input_dim = attack_steps + services
	else:
	 	input_dim = attack_steps

	agent = ReinforceAgent(input_dim, services, hidden_dim=args.hidden_width, allow_skip=allow_skip)

	#Train
	run_multiple_simulations(args.n_simulations, env, agent)

	#Evaluate
	if args.evaluation_rounds > 0:
		run_multiple_simulations(args.evaluation_rounds, env, agent, evaluation=True, include_services=include_services_in_state)

	if args.test:
		test_correctness(env, agent, graph_size=args.graph_size, include_services=include_services_in_state)

