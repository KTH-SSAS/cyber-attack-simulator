from attack_simulator.agents.tabular_agents import Agent
import torch
import logging


class OptimalAgent(Agent):
    # Not really optimal (yet). Fails to consider AND steps and does not take any chances even if parent steps to valuables have long ttc's.
    def __init__(self, env) -> None:
        self.attack_graph = env.attack_graph
        self.attacker = env.attacker
        self.can_skip = True
        self.n_action = 0
        self.previous_state = [False]*len(self.attack_graph.attack_steps)

    def act(self, state):
        log = logging.getLogger("simulator")
        action_id = 0
        # If an attack step has been compromised which has a valuable child, then disable the corresponding service.
        for step_id in range(0, len(state)):
            if state[step_id]:
                step_name = list(self.attack_graph.attack_steps)[step_id]
                for child_name in self.attack_graph.attack_steps[step_name].children:
                    if self.attack_graph.attack_steps[child_name].reward > 0:
                        service = self.corresponding_service(step_name)
                        if self.attack_graph.enabled_services[service]:
                            # action_id + 1 because action == 0 is no action.
                            action_id = list(self.attack_graph.enabled_services).index(service) + 1
        # If no service should be disabled, then return 0
        self.previous_state = state
        return action_id

    def corresponding_service(self, attack_step_name):
        for service_name in self.attack_graph.enabled_services:
            if self.attack_graph.attack_steps[attack_step_name].asset in service_name:
                if self.attack_graph.attack_steps[attack_step_name].service == '':
                    return service_name
                elif self.attack_graph.attack_steps[attack_step_name].service in service_name:
                    return service_name
        assert(False, "Attack step doesn't correspond to any service.")
                

    def update(self, rewards):
        return torch.Tensor([0])

    def calculate_loss(self, rewards, normalize_returns=False):
        return torch.Tensor([0])

    def eval(self):
        pass

    def train(self):
        pass
