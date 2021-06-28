from attack_simulator.attack_simulation_env import AttackSimulationEnv
from attack_simulator.tabular_agents import Agent
import unittest
import matplotlib.pyplot as plt
import logging
import numpy.random as random

def bool2int(x):
	y = 0
	for i, j in enumerate(x):
		y += j<<i
	return y

def run_sim(env: AttackSimulationEnv, agent: Agent, plot_results=False):
	enabled_services: dict = {}

	#twin1.spines.right.set_position(("axes", 1.2))
	log= logging.getLogger("simulator")
	done = False
	for service in env.attack_graph.enabled_services:
		enabled_services[service] = 1

	rewards = []
	num_services = []
	state = env._next_observation() # Intial state
	while not done:
		state_idx = bool2int(state)
		action = agent.act(state_idx)
		enabled_services[action] = False
		new_state, reward, done, info = env.step(tuple(enabled_services.values()))
		agent.update(bool2int(new_state), reward)
		rewards.append(reward)
		num_services.append(sum(list(enabled_services.values())))

		if info["time_on_current_step"] == 1:
			log.debug(str(info['time']) + ": reward=" + str(reward) + ". Attacking " + str(info['current_step']))

	if plot_results:	
		_, ax = plt.subplots()
		twin1 = ax.twinx()
		ax.plot(rewards, "b")
		twin1.plot(num_services, "r")
		plt.show()

	log.debug(f"Return: {sum(rewards)}")
	return sum(rewards)

def run_multiple_simulations(episodes, env: AttackSimulationEnv, agent: Agent):

	returns = []
	for _ in range(episodes):
		sum_reward = run_sim(env, agent)
		returns.append(sum_reward)
		env.reset()
	
	plt.plot(returns)
	plt.show()


class TestAgents(unittest.TestCase):

	def setUp(self):
		from attack_simulator.attack_simulation_env import AttackSimulationEnv
		random.seed(666)
		logging.basicConfig(filename='simulator.log')
		logging.getLogger("simulator").setLevel(logging.DEBUG)
		self.env=AttackSimulationEnv()

	def test_random_agent(self):
		from attack_simulator.tabular_agents import RandomAgent
		agent = RandomAgent(0.001)
		run_sim(self.env, agent)

	def test_bandit(self):
		from attack_simulator.tabular_agents import BanditAgent
		agent = BanditAgent(self.env.n_defender_actions)

		run_sim(self.env, agent)

	def test_q_learner(self):
		from attack_simulator.tabular_agents import QLearningAgent
		agent = QLearningAgent(self.env.n_defender_actions)
		run_multiple_simulations(1000, self.env, agent)

if __name__ == '__main__':
	unittest.main()
