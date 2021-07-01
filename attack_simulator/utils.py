from attack_simulator.policy_agents import ReinforceAgent
from attack_simulator.attack_simulation_env import AttackSimulationEnv
import matplotlib.pyplot as plt
import logging
import numpy.random as random
import numpy as np

def run_sim(env: AttackSimulationEnv, agent: ReinforceAgent, plot_results=False):
	enabled_services: dict = {}
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
		new_state, reward, done, info = env.step(tuple(enabled_services.values()))
		rewards.append(reward)
		# count number of running services
		num_services.append(sum(list(enabled_services.values())))
		state = new_state
		#if info["time_on_current_step"] == 1:
		#	logger.debug(str(info['time']) + ": reward=" + str(reward) +
		#			  ". Attacking " + str(info['current_step']))


	if plot_results:
		_, ax = plt.subplots()
		twin1 = ax.twinx()
		ax.plot(rewards, "b")
		twin1.plot(num_services, "r")
		plt.show()
	
	return rewards, info['time']


def run_multiple_simulations(episodes, env: AttackSimulationEnv, agent: ReinforceAgent):

	log = logging.getLogger("trainer")
	returns = np.zeros(episodes)
	losses = np.zeros(episodes)
	lengths = np.zeros(episodes)
	try:
		for i in range(episodes):
			rewards, episode_length = run_sim(env, agent)
			loss = agent.update(rewards)
			losses[i] = loss
			returns[i] = sum(rewards)
			lengths[i] = episode_length
			env.reset()		
			log.debug(f"Episode: {i+1}/{episodes}, Return: {sum(rewards)}, Episode Length: {episode_length}")
	except KeyboardInterrupt:
		print("Stopping...")
	fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
	ax1.plot(returns)
	#ax1.set_xlabel("Episode")
	ax1.set_xlim(0, i) # Cut off graph at stopping point
	ax1.set_ylabel("Return")
	ax2.plot(losses)
	ax2.set_ylabel('Loss')
	#ax2.set_xlabel('Episode')
	ax3.plot(lengths)
	ax3.set_xlabel("Episode")
	ax3.set_ylabel("Episode Length")
	fig.savefig('plot.pdf', dpi=200)
	plt.show()
