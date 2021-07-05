from typing import Dict, List
import gym
from gym import spaces
import numpy as np
import random
from attack_simulator.attack_graph import AttackGraph, AttackStep
import logging

# The probability that the defender will disable a given service at a given step is given by DISABLE_PROBABILITY.
DISABLE_PROBABILITY = 0.001
# For debugging convenience, simulations can be made deterministic, only dependent on the random seed.
DETERMINISTIC = False
RANDOM_SEED = 4

class Attacker:
    
    def __init__(self, attack_graph: AttackGraph, compromised_steps: List[str], deterministic=False):
        self.attack_graph = attack_graph
        # self.compromised_steps keeps track of which attack steps have been reached by that attacker.
        self.compromised_steps = compromised_steps
        self.deterministic = deterministic
        self.choose_next_step()
        self.time_on_current_step = 0
        self.total_time = 0

    def get_step(self, name) -> AttackStep:
        return self.attack_graph.attack_steps[name]
    
    def attack_surface(self, debug=False):
        # The attack surface consists of all reachable but uncompromised attack steps.
        att_surf = set()
        for compromised_step_name in self.compromised_steps:
            for child_name in self.get_step(compromised_step_name).children:
                if self.get_step(child_name).enabled:
                    if self.get_step(child_name).step_type == 'or':
                        att_surf.add(child_name)
                    else:
                        all_parents_are_compromised = True
                        for parent_name in self.get_step(child_name).parents:
                            if parent_name not in self.compromised_steps:
                                all_parents_are_compromised = False
                                break
                        if all_parents_are_compromised:
                            att_surf.add(child_name)

        att_surf -= set(self.compromised_steps)
        return att_surf

    def choose_next_step(self): 
        # The attacker strategy is currently simply to select a random attack step of the available ones (i.e. from the attack surface).
        self.current_step = None
        if self.attack_surface():
            if self.deterministic:
                self.current_step = sorted(list(self.attack_surface()))[0]
            else:
                self.current_step = random.choice(list(self.attack_surface()))

    def attack(self):
        # If the attacker has run out of attack steps, then terminate.
        if not self.current_step:
            return False
        self.reward = 0
        # If the attacker has spent the required time on the current attack step, then it becomes compromised.
        if self.time_on_current_step >= self.get_step(self.current_step).ttc:
            self.compromised_steps.append(self.current_step)
            self.reward = self.attack_graph.attack_steps[self.current_step].reward
            # print(f"Compromised attack step with reward {self.reward}")
            # If the attack surface (the available uncompromised attack steps) is empty, then terminate.
            if not self.attack_surface():
                return False
            self.choose_next_step()
            self.time_on_current_step = 0
        # Keep track of the time spent.
        self.time_on_current_step += 1
        self.total_time += 1
        return True

    def observe(self, attack_step):
        # Observations of the attacker are made by an intrusion detection system. 
        # The accuracy of observations is given for each attack step by the true and false positive rates respectively.
        if self.deterministic:
            return attack_step in self.compromised_steps
        else:
            rnd = random.uniform(0,1)
            if attack_step in self.compromised_steps:
                return rnd <= self.get_step(attack_step).true_positive
            else:
                return rnd <= self.get_step(attack_step).false_positive


class AttackSimulationEnv(gym.Env):

    def __init__(self, deterministic=False, flag_reward=1000):
        super(AttackSimulationEnv, self).__init__()
        self.deterministic = deterministic
        self.flag_reward = flag_reward
        print(f"AttackSimulationEnv's flag reward is {self.flag_reward}")
        self.attack_graph = AttackGraph(deterministic=deterministic, flag_reward=flag_reward)
        self.attacker = Attacker(self.attack_graph, ['internet.connect'], deterministic=deterministic)
        # An observation informs the defender of which attack steps have been compromised.
        # Observations are imperfect.
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.attack_graph.size, 1), dtype=np.float32)
        # The defender action space consists of the disablement of services and hosts.        
        self.n_defender_actions = len(self.attack_graph.enabled_services)        
        self.action_space = spaces.Tuple(([spaces.Discrete(2)]*self.n_defender_actions))

    def get_info(self):
        if self.attacker.current_step:
            info = {"time": self.attacker.total_time, "current_step": self.attacker.current_step, "time_on_current_step": self.attacker.time_on_current_step, "ttc_of_current_step": self.attacker.get_step(self.attacker.current_step).ttc, "attack_surface": self.attacker.attack_surface(), "self.attack_graph.enabled_services": self.attack_graph.enabled_services}
        else:
            info = {"time": self.attacker.total_time, "current_step": None, "time_on_current_step": None, "ttc_of_current_step": None, "attack_surface": self.attacker.attack_surface(), "self.attack_graph.enabled_services": self.attack_graph.enabled_services}
        return info


    def step(self, action):
        logger = logging.getLogger("simulator")
        # The order of actions follows self.attack_graph.enabled_services
        action_id = 0
        # provision_reward is the defender reward for maintaining services online. 
        self.provision_reward = 0
        # Disable services according to the actions provided
        for service in self.attack_graph.enabled_services:
            if self.attack_graph.enabled_services[service]:
                self.provision_reward += 1
                if action[action_id] == 0:
                    self.disable(service)
            action_id += 1

        obs = self._next_observation()
        # The attacker attacks. If the attacker's attack surface is empty, then the game ends.
        attacker_done = not self.attacker.attack()

        # Positive rewards for maintaining services enabled_services and negative for compromised flags.
        reward = self.provision_reward - self.attacker.reward
        info = self.get_info()
        logger.debug(str(info['time']) + ": reward=" + str(reward) + ". Attacking " + str(info['current_step']))
        if attacker_done:
            logger.debug("Attacker is done.")
            logger.debug(f"Compromised steps: {self.attacker.compromised_steps}")
        return obs, reward, attacker_done, info

    def reset(self):
        logger = logging.getLogger("simulator")
        logger.debug("Starting new simulation.")
        logger.debug(f"self.flag_reward = {self.flag_reward}")
        self.attack_graph.reset(deterministic=self.deterministic, flag_reward=self.flag_reward)
        self.attacker = Attacker(self.attack_graph, ['internet.connect'])
        return self._next_observation()

    def _next_observation(self):
        # Imperfect observations by intrusion detection system
        return np.array([self.attacker.observe(a) for a in self.attack_graph.attack_steps])

    def render(self, mode='human'):
        pass

    def disable(self, service):
        logger = logging.getLogger('simulator')
        if self.attack_graph.enabled_services[service]:
                logger.debug(f"Disabling {service}")
        self.attack_graph.disable(service)
        self.attacker.choose_next_step()


if __name__ == '__main__':
    if DETERMINISTIC:
        random.seed(RANDOM_SEED)

    env = AttackSimulationEnv()
    enabled_services = dict()
    # Defender can act by disabling various services and hosts (found in env.attack_graph.enabled_services)
    for service in env.attack_graph.enabled_services:
        enabled_services[service] = 1
    done = False
    while not done:
        enabled_services_status_changed = False
        for service in enabled_services:
            # Current strategy is to disable any service with a given probability each step.
            if enabled_services[service] == 1 and random.uniform(0,1) < DISABLE_PROBABILITY:
                # print("Defender disabling " + service)
                enabled_services[service] = 0
                enabled_services_status_changed = True
        obs, reward, done, info = env.step(tuple(enabled_services.values()))
        # if info["time_on_current_step"] == 1 or enabled_services_status_changed:
            # print(str(info['time']) + ": reward=" + str(reward) + ". Attacking " + str(info['current_step']))
