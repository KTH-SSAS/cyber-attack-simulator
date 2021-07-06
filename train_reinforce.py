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
	parser.add_argument('-t', '--test', action='store_true', help='Run tests.')
	args = parser.parse_args()

	random.seed(0)
	torch.manual_seed(0)
	logging.getLogger("simulator").setLevel(logging.DEBUG)
	logging.getLogger("simulator").addHandler(logging.FileHandler("simulator.log", mode="w"))
	logging.getLogger("trainer").setLevel(logging.DEBUG)
	logging.getLogger("trainer").addHandler(logging.FileHandler("trainer.log", mode="w"))
	env = AttackSimulationEnv(deterministic=False, flag_reward=100000)
	attack_steps = 78
	services = 17
	agent = ReinforceAgent(attack_steps, services, hidden_dim=64)

	if args.test:
		run_multiple_simulations(10, env, agent)
		test(env, agent)
	else:
		run_multiple_simulations(100000, env, agent)

