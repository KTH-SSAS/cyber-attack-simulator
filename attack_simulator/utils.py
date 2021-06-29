from attack_simulator.policy_agents import ReinforceAgent
from attack_simulator.attack_simulation_env import AttackSimulationEnv
import matplotlib.pyplot as plt
import logging
import numpy.random as random

def run_sim(env: AttackSimulationEnv, agent: ReinforceAgent, plot_results=False):
	enabled_services: dict = {}
	logger = logging.getLogger("simulator")
	#twin1.spines.right.set_position(("axes", 1.2))
	done = False
	for service in env.attack_graph.enabled_services:
		enabled_services[service] = 1

	rewards = []
	num_services = []
	state = env._next_observation()  # Intial state
	while not done:
		action = agent.act(state)
		for i, key in enumerate(enabled_services):
			if i == action:
				enabled_services[key] = 0
				break
		_, reward, done, _ = env.step(tuple(enabled_services.values()))
		rewards.append(reward)
		# count number of running services
		num_services.append(sum(list(enabled_services.values())))

		#if info["time_on_current_step"] == 1:
		#	logger.debug(str(info['time']) + ": reward=" + str(reward) +
		#			  ". Attacking " + str(info['current_step']))

	if plot_results:
		_, ax = plt.subplots()
		twin1 = ax.twinx()
		ax.plot(rewards, "b")
		twin1.plot(num_services, "r")
		plt.show()

	
	return rewards


def run_multiple_simulations(episodes, env: AttackSimulationEnv, agent: ReinforceAgent):

	log = logging.getLogger("trainer")
	returns = []
	losses = []
	for i in range(episodes):
		rewards = run_sim(env, agent)
		loss = agent.update(rewards)
		losses.append(loss)
		returns.append(sum(rewards))
		env.reset()		
		log.debug(f"Episode: {i+1}/{episodes}, Return: {sum(rewards)}")

	plt.plot(returns)
	plt.xlabel("Episode")
	plt.ylabel("Return")
	plt.show()
	plt.plot(losses)
	plt.ylabel('Loss')
	plt.xlabel('Episode')
	plt.show()