import logging
from attack_simulator.attack_simulation_env import AttackSimulationEnv

def test(env, agent):
	# Testing a specific state action. If 'lazarus.ftp.login', 'lazarus.ftp.dictionary_attack', 'lazarus.ftp.connect', 'office_network.map', 'office_network.connect' and 'internet.connect' are compromised, then the best action must be to disable 'lazarus'.
	log = logging.getLogger("trainer")
	compromised_steps = ['lazarus.ftp.login', 'lazarus.ftp.dictionary_attack', 'lazarus.ftp.connect', 'office_network.map', 'office_network.connect', 'internet.connect']
	correct_action = 'lazarus'
	state = env.observation_from_compromised_steps(compromised_steps)
	action = env.interpret_action(agent.act(state))
	if action == correct_action: 
		log.debug("Test succeeded: Selected correct action in test lazarus.")
	else:   
		log.debug(f"Test failed: Selected action {action} instead of {correct_action}.")

