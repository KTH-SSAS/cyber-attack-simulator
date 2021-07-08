from attack_simulator.policy_agents import ReinforceAgent
import logging
import torch
from torch.distributions import Categorical
from attack_simulator.attack_simulation_env import AttackSimulationEnv

def test_state_action_response(env: AttackSimulationEnv, agent: ReinforceAgent, test_no, compromised_steps, correct_action):
	log = logging.getLogger("trainer")
	state = env.observation_from_compromised_steps(compromised_steps)
	action_probabilities = agent.policy.forward(torch.Tensor(state))
	action_id = agent.act(state)
	action = env.interpret_action(action_id)
	if action in correct_action: 
		log.debug(f"Test {test_no} succeeded: Selected correct action {action}.")
	else:   
		log.debug(f"Test {test_no} failed: Selected action {action} instead of {correct_action}.")


def test_correctness(env, agent: ReinforceAgent, graph_size='large'):
	agent.eval()
	# Testing a specific state action. If 'lazarus.ftp.login', 'lazarus.ftp.dictionary_attack', 'lazarus.ftp.connect', 'office_network.map', 'office_network.connect' and 'internet.connect' are compromised, then the best action must be to disable 'lazarus'.
	compromised_steps = []
	correct_actions = []
	#compromised_steps.append(['lazarus.ftp.login', 'lazarus.ftp.dictionary_attack', 'lazarus.ftp.connect', 'office_network.map', 'office_network.connect', 'internet.connect'])
	compromised_steps.append(['lazarus.ftp.dictionary_attack', 'lazarus.ftp.connect', 'office_network.map', 'office_network.connect', 'internet.connect'])
	correct_actions.append(['lazarus', 'lazarus.ftp'])

	if graph_size == 'large' or graph_size == 'medium':
		compromised_steps.append(['lazarus.terminal_access', 'lazarus.tomcat.exploit_vulnerability', 'lazarus.tomcat.find_vulnerability', 'lazarus.tomcat.dictionary_attack', 'lazarus.tomcat.gather_information', 'lazarus.tomcat.connect', 'office_network.map', 'office_network.connect', 'internet.connect'])
		correct_actions.append(['lazarus'])

		compromised_steps.append(['energetic_bear.escalate_to_root', 'energetic_bear.exploit_vulnerability', 'energetic_bear.find_vulnerability', 'energetic_bear.terminal_access', 'energetic_bear.apache.exploit_vulnerability', 'energetic_bear.apache.find_vulnerability', 'energetic_bear.apache.gather_information', 'energetic_bear.apache.connect', 'office_network.map', 'office_network.connect', 'internet.connect'])
		correct_actions.append(['energetic_bear'])

	for i in range(0,len(compromised_steps)):
		test_state_action_response(env, agent, i, compromised_steps[i], correct_actions[i])
