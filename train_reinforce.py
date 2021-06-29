from attack_simulator.utils import run_multiple_simulations
from attack_simulator.policy_agents import ReinforceAgent
from attack_simulator.attack_simulation_env import AttackSimulationEnv
import logging
import numpy.random as random
import torch

if __name__ == '__main__':
	random.seed(0)
	torch.manual_seed(0)
	logging.getLogger("simulator").setLevel(logging.DEBUG)
	logging.getLogger("simulator").addHandler(logging.FileHandler("simulator.log", mode="w"))
	logging.getLogger("trainer").setLevel(logging.DEBUG)
	logging.getLogger("trainer").addHandler(logging.FileHandler("trainer.log", mode="w"))
	env = AttackSimulationEnv()
	agent = ReinforceAgent(78, 16, 64)
	run_multiple_simulations(5000, env, agent)