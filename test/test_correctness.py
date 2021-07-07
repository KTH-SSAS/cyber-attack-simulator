import logging
from attack_simulator.attack_simulation_env import AttackSimulationEnv

def test_state_action_response(env, agent, test_no, compromised_steps, correct_action):
	log = logging.getLogger("trainer")
	state = env.observation_from_compromised_steps(compromised_steps)
	action = env.interpret_action(agent.act(state))
	if action == correct_action: 
		log.debug(f"Test {test_no} succeeded: Selected correct action in test lazarus.")
	else:   
		log.debug(f"Test {test_no} failed: Selected action {action} instead of {correct_action}.")


def test(env, agent, graph_size='large'):
	# Testing a specific state action. If 'lazarus.ftp.login', 'lazarus.ftp.dictionary_attack', 'lazarus.ftp.connect', 'office_network.map', 'office_network.connect' and 'internet.connect' are compromised, then the best action must be to disable 'lazarus'.
	compromised_steps = []
	correct_actions = []
	compromised_steps.append(['lazarus.ftp.login', 'lazarus.ftp.dictionary_attack', 'lazarus.ftp.connect', 'office_network.map', 'office_network.connect', 'internet.connect'])
	correct_actions.append('lazarus')

	if graph_size == 'large':
		compromised_steps.append(['lazarus.terminal_access', 'lazarus.tomcat.exploit_vulnerability', 'lazarus.tomcat.find_vulnerability', 'lazarus.tomcat.dictionary_attack', 'lazarus.tomcat.gather_information', 'lazarus.tomcat.connect', 'office_network.map', 'office_network.connect', 'internet.connect'])
		correct_actions.append('lazarus')

		compromised_steps.append(['energetic_bear.escalate_to_root', 'energetic_bear.exploit_vulnerability', 'energetic_bear.find_vulnerability', 'energetic_bear.terminal_access', 'energetic_bear.apache.exploit_vulnerability', 'energetic_bear.apache.find_vulnerability', 'energetic_bear.apache.gather_information', 'energetic_bear.apache.connect', 'office_network.map', 'office_network.connect', 'internet.connect'])
		correct_actions.append('energetic_bear')

	for i in range(0,len(compromised_steps)):
		test_state_action_response(env, agent, i, compromised_steps[i], correct_actions[i])
