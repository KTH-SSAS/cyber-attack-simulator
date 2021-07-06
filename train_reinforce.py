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
	parser.add_argument('-t', '--test', action='store_true', help='Run tests.')
	parser.add_argument('-s', '--simulations', type=int, default=100, help='Number of simulations.')
	parser.add_argument('-f', '--flag_rewards', type=int, default=1000, help='Flag rewards for the attacker (use positive values).')
	args = parser.parse_args()

	random.seed(0)
	torch.manual_seed(0)
	logging.getLogger("simulator").setLevel(logging.DEBUG)
	logging.getLogger("simulator").addHandler(logging.FileHandler("simulator.log", mode="w"))
	logging.getLogger("trainer").setLevel(logging.DEBUG)
	logging.getLogger("trainer").addHandler(logging.FileHandler("trainer.log", mode="w"))
	env = AttackSimulationEnv(deterministic=False, flag_reward=args.flag_rewards)
	if args.graph:
		env.attack_graph.generate_graphviz_file()
	attack_steps = 78
	services = 17
	agent = ReinforceAgent(attack_steps, services, hidden_dim=64)

	run_multiple_simulations(args.simulations, env, agent)
	if args.test:
		test(env, agent)

