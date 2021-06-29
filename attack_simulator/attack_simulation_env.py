from typing import Dict, List
import gym
from gym import spaces
import numpy as np
import random
import logging

# The probability that the defender will disable a given service at a given step is given by DISABLE_PROBABILITY.
DISABLE_PROBABILITY = 0.001
# For debugging convenience, simulations can be made deterministic, only dependent on the random seed.
DETERMINISTIC = False
RANDOM_SEED = 4

class AttackStep:

    def __init__(self, name='', step_type='or', ttc=1, reward=0, children={}, true_positive=0.95, false_positive=0.1):
        self.name = name
        self.step_type = step_type
        if DETERMINISTIC:
            self.ttc = ttc
            self.reward = reward
        else:
            self.ttc = int(np.random.exponential(scale=ttc))
            self.reward = int(np.random.exponential(scale=reward))
        self.children = children
        self.parents = set()
        self.true_positive = true_positive
        self.false_positive = false_positive
        self.enabled = True


class AttackGraph:

    def __init__(self):
        self.reset()

    def reset(self):
        logger = logging.getLogger("simulator")
        # These are the services and hosts that the defender is at liberty to disable in order to protect the computer network.
        self.enabled_services = dict()
        # Disabling a host, e.g. lazarus, will also disable all of its services
        self.enabled_services['lazarus'] = True
        # The defender can limit disablement to a single service, such as the ftp server on lazarus
        self.enabled_services['lazarus.ftp'] = True
        self.enabled_services['lazarus.tomcat'] = True
        self.enabled_services['energetic_bear'] = True
        self.enabled_services['energetic_bear.apache'] = True
        self.enabled_services['sea_turtle'] = True
        self.enabled_services['sea_turtle.telnet'] = True
        self.enabled_services['buckeye'] = True
        self.enabled_services['buckeye.firefox'] = True
        self.enabled_services['cloud_hopper'] = True
        self.enabled_services['cloud_hopper.smb'] = True
        self.enabled_services['cloud_function'] = True
        self.enabled_services['fancy_bear.ssh'] = True
        self.enabled_services['fancy_bear'] = True
        self.enabled_services['wifi_host.http_server'] = True
        self.enabled_services['wifi_host'] = True
        
        logger.info(str(len(self.enabled_services)) + " possible defender actions.")

        self.attack_steps: Dict[str, AttackStep] = {}

        # Here the attack logic is defined. The below is a model of the EN2720 course.
        self.attack_steps['wifi_host.http_server.flag_18dd8f.capture'] = AttackStep(reward=1000)
        self.attack_steps['wifi_host.http_server.gather_information'] = AttackStep(ttc=3, children={'wifi_host.http_server.flag_18dd8f.capture'})
        self.attack_steps['wifi_host.http_server.connect'] = AttackStep(children={'wifi_host.http_server.gather_information'})

        self.attack_steps['wifi_network.map'] = AttackStep(ttc=10, children={'wifi_host.http_server.connect'})
        self.attack_steps['wifi_network.connect'] = AttackStep(step_type='and', children={'wifi_network.map'})
        self.attack_steps['wifi_network.flag_d582aa.capture'] = AttackStep(reward=1000)
        self.attack_steps['wifi_network.decrypt_traffic'] = AttackStep(step_type='and', ttc=5, children={'wifi_network.flag_d582aa.capture'})
        self.attack_steps['wifi_network.obtain_credentials'] = AttackStep(ttc=10, children={'wifi_network.connect', 'wifi_network.decrypt_traffic'})
        self.attack_steps['wifi_network.find_credentials'] = AttackStep(children={'wifi_network.obtain_credentials'})
        self.attack_steps['wifi_network.capture_traffic'] = AttackStep(ttc=5, children={'wifi_network.connect', 'wifi_network.find_credentials', 'wifi_network.decrypt_traffic'})

        self.attack_steps['fancy_bear.gather_information'] = AttackStep(ttc=5, children={'wifi_network.capture_traffic'})
        
        self.attack_steps['fancy_bear.ssh.login'] = AttackStep(step_type='and', children={'fancy_bear.gather_information'})
        self.attack_steps['fancy_bear.ssh.obtain_credentials'] = AttackStep(children={'fancy_bear.ssh.login'})
        self.attack_steps['fancy_bear.ssh.connect'] = AttackStep(children={'fancy_bear.ssh.login'})
        
        self.attack_steps['cloud_function.flag_831865.capture'] = AttackStep(reward=1000)
        self.attack_steps['cloud_function.flag_d8d9da.capture'] = AttackStep(reward=1000)
        self.attack_steps['cloud_function.exploit_vulnerability'] = AttackStep(children={'cloud_function.flag_831865.capture'})
        self.attack_steps['cloud_function.find_vulnerability'] = AttackStep(children={'cloud_function.exploit_vulnerability'})
        self.attack_steps['cloud_function.gather_information'] = AttackStep(children={'cloud_function.find_vulnerability', 'cloud_function.flag_d8d9da.capture'})

        self.attack_steps['cloud_bucket.flag_21077e.capture'] = AttackStep(reward=1000)
        self.attack_steps['cloud_bucket.find_credentials'] = AttackStep(children={'fancy_bear.ssh.obtain_credentials'})
        self.attack_steps['cloud_bucket.list'] = AttackStep(children={'cloud_function.gather_information', 'cloud_bucket.find_credentials', 'cloud_bucket.flag_21077e.capture'})

        self.attack_steps['cloud_hopper.gather_information'] = AttackStep(ttc=5, children={'cloud_bucket.list'})
        self.attack_steps['cloud_hopper.flag_93b00a.capture'] = AttackStep(reward=1000)
        self.attack_steps['cloud_hopper.terminal_access'] = AttackStep(children={'cloud_hopper.gather_information', 'cloud_hopper.flag_93b00a.capture'})
        self.attack_steps['cloud_hopper.smb.exploit_vulnerability'] = AttackStep(ttc=10, children={'cloud_hopper.terminal_access'})
        self.attack_steps['cloud_hopper.smb.find_vulnerability'] = AttackStep(ttc=5, children={'cloud_hopper.smb.exploit_vulnerability'})
        self.attack_steps['cloud_hopper.smb.connect'] = AttackStep(children={'cloud_hopper.smb.find_vulnerability'})
        
        self.attack_steps['hidden_network.map'] = AttackStep(ttc=10, children={'cloud_hopper.smb.connect', 'fancy_bear.ssh.connect'})
        self.attack_steps['hidden_network.connect'] = AttackStep(children={'hidden_network.map'})

        self.attack_steps['buckeye.flag_2362e5.capture'] = AttackStep(reward=1000)
        self.attack_steps['buckeye.flag_5d402e.capture'] = AttackStep(reward=1000)
        self.attack_steps['buckeye.firefox.flag_14ce18.capture'] = AttackStep(reward=1000)
        self.attack_steps['buckeye.escalate_to_root'] = AttackStep(children={'hidden_network.connect', 'buckeye.flag_2362e5.capture'})
        self.attack_steps['buckeye.exploit_vulnerability'] = AttackStep(ttc=10, children={'buckeye.escalate_to_root'})
        self.attack_steps['buckeye.find_vulnerability'] = AttackStep(ttc=5, children={'buckeye.exploit_vulnerability'})
        self.attack_steps['buckeye.terminal_access'] = AttackStep(children={'buckeye.find_vulnerability', 'buckeye.flag_5d402e.capture'})
        self.attack_steps['buckeye.firefox.exploit_vulnerability'] = AttackStep(ttc=10, children={'buckeye.terminal_access'})
        self.attack_steps['buckeye.firefox.find_vulnerability'] = AttackStep(ttc=5, children={'buckeye.firefox.exploit_vulnerability'})
        self.attack_steps['buckeye.firefox.connect'] = AttackStep(children={'buckeye.firefox.find_vulnerability'})

        self.attack_steps['energetic_bear.flag_73cb43.capture'] = AttackStep(reward=1000)
        self.attack_steps['energetic_bear.flag_3b2000.capture'] = AttackStep(reward=1000)
        self.attack_steps['energetic_bear.flag_de3b1c.capture'] = AttackStep(reward=1000)
        self.attack_steps['energetic_bear.capture_traffic'] = AttackStep(ttc=5, children={'buckeye.firefox.connect', 'buckeye.firefox.flag_14ce18.capture'})
        self.attack_steps['energetic_bear.escalate_to_root'] = AttackStep(ttc=20, children={'energetic_bear.capture_traffic', 'energetic_bear.flag_73cb43.capture'})
        self.attack_steps['energetic_bear.exploit_vulnerability'] = AttackStep(ttc=50, children={'energetic_bear.escalate_to_root'})
        self.attack_steps['energetic_bear.find_vulnerability'] = AttackStep(ttc=50, children={'energetic_bear.exploit_vulnerability'})
        self.attack_steps['energetic_bear.terminal_access'] = AttackStep(ttc=10, children={'energetic_bear.find_vulnerability', 'energetic_bear.flag_3b2000.capture'})
        
        self.attack_steps['energetic_bear.apache.flag_521bce.capture'] = AttackStep(reward=1000)
        self.attack_steps['energetic_bear.apache.exploit_vulnerability'] = AttackStep(ttc=20, children={'energetic_bear.terminal_access', 'energetic_bear.flag_de3b1c.capture'})
        self.attack_steps['energetic_bear.apache.find_vulnerability'] = AttackStep(ttc=30, children={'energetic_bear.apache.exploit_vulnerability'})
        self.attack_steps['energetic_bear.apache.gather_information'] = AttackStep(ttc=10, children={'energetic_bear.apache.find_vulnerability', 'energetic_bear.apache.flag_521bce.capture'})
        self.attack_steps['energetic_bear.apache.connect'] = AttackStep(children={'energetic_bear.apache.gather_information'})

        self.attack_steps['sea_turle.flag_6be6ef.capture'] = AttackStep(reward=1000)
        self.attack_steps['sea_turle.flag_f9038f.capture'] = AttackStep(reward=1000)
        self.attack_steps['sea_turtle.capture_traffic'] = AttackStep(ttc=10, children={'buckeye.firefox.connect', 'buckeye.firefox.flag_14ce18.capture'})
        self.attack_steps['sea_turtle.escalate_to_root'] = AttackStep(ttc=50, children={'sea_turle.flag_6be6ef.capture'})
        self.attack_steps['sea_turtle.exploit_vulnerability'] = AttackStep(ttc=50, children={'sea_turtle.escalate_to_root'})
        self.attack_steps['sea_turtle.find_vulnerability'] = AttackStep(ttc=50, children={'sea_turtle.exploit_vulnerability'})
        self.attack_steps['sea_turtle.telnet.login'] = AttackStep(step_type='and', children={'sea_turtle.find_vulnerability', 'sea_turle.flag_f9038f.capture'})
        self.attack_steps['sea_turtle.telnet.obtain_credentials'] = AttackStep(children={'sea_turtle.telnet.login'})
        self.attack_steps['sea_turtle.telnet.connect'] = AttackStep(children={'sea_turtle.telnet.login'})

        self.attack_steps['lazarus.flag_cd699a.capture'] = AttackStep(reward=1000)
        self.attack_steps['lazarus.find_credentials'] = AttackStep(ttc=10, children={'sea_turtle.telnet.obtain_credentials'})
        self.attack_steps['lazarus.terminal_access'] = AttackStep(children={'lazarus.find_credentials', 'lazarus.flag_cd699a.capture'})
        
        self.attack_steps['lazarus.tomcat.flag_90b353.capture'] = AttackStep(reward=1000)
        self.attack_steps['lazarus.tomcat.exploit_vulnerability'] = AttackStep(ttc=10, children={'lazarus.terminal_access'})
        self.attack_steps['lazarus.tomcat.find_vulnerability'] = AttackStep(ttc=10, children={'lazarus.tomcat.exploit_vulnerability'})
        self.attack_steps['lazarus.tomcat.dictionary_attack'] = AttackStep(ttc=10, children={'lazarus.tomcat.find_vulnerability', 'lazarus.tomcat.flag_90b353.capture'})
        self.attack_steps['lazarus.tomcat.gather_information'] = AttackStep(ttc=5, children={'lazarus.tomcat.dictionary_attack'})
        self.attack_steps['lazarus.tomcat.connect'] = AttackStep(children={'lazarus.tomcat.gather_information'})

        self.attack_steps['lazarus.ftp.flag_adcb1f.capture'] = AttackStep(reward=1000)
        self.attack_steps['lazarus.ftp.login'] = AttackStep(children={'lazarus.ftp.flag_adcb1f.capture'})
        self.attack_steps['lazarus.ftp.dictionary_attack'] = AttackStep(ttc=100, children={'lazarus.ftp.login'})
        self.attack_steps['lazarus.ftp.connect'] = AttackStep(children={'lazarus.ftp.dictionary_attack'})

        self.attack_steps['office_network.map'] = AttackStep(ttc=10, children={'lazarus.ftp.connect', 'energetic_bear.apache.connect', 'lazarus.tomcat.connect', 'sea_turtle.telnet.connect'})
        self.attack_steps['office_network.connect'] = AttackStep(children={'office_network.map'})
        self.attack_steps['internet.connect'] = AttackStep(children={'office_network.connect'})
        
        self.record_parents()
        self.size = len(self.attack_steps)

        logger.info(str(self.size) + " attack steps.")

    def record_parents(self):
        #And steps need to know which their parents are.
        for parent in self.attack_steps:
            for child in self.attack_steps[parent].children:
                self.attack_steps[child].parents.add(parent)

    def disable(self, service):
        logger = logging.getLogger('simulator')
        # Disconnect all attack steps that match the service.
        for step_name in self.attack_steps:
            if self.attack_steps[step_name].enabled and service in step_name:
                self.attack_steps[step_name].enabled = False
        # Also disable subservices (e.g. fancy_bear.ssh should be disabled when fancy_bear is)
        for subservice in self.enabled_services:
            if self.enabled_services[subservice]: # service is enabled
                if service in subservice and service != subservice:
                    self.enabled_services[subservice] = False
                    logger.debug("Disabling subservice " + subservice)


class Attacker:
    
    def __init__(self, attack_graph: AttackGraph, compromised_steps: List[str]):
        self.attack_graph = attack_graph
        # self.compromised_steps keeps track of which attack steps have been reached by that attacker.
        self.compromised_steps = compromised_steps
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
            if DETERMINISTIC:
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
        if DETERMINISTIC:
            return attack_step in self.compromised_steps
        else:
            rnd = random.uniform(0,1)
            if attack_step in self.compromised_steps:
                return rnd <= self.get_step(attack_step).true_positive
            else:
                return rnd <= self.get_step(attack_step).false_positive


class AttackSimulationEnv(gym.Env):

    def __init__(self):
        super(AttackSimulationEnv, self).__init__()
        self.attack_graph = AttackGraph()
        self.attacker = Attacker(self.attack_graph, ['internet.connect'])
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
        # Isolate services according to the actions provided
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
        return obs, reward, attacker_done, info

    def reset(self):
        logger = logging.getLogger("simulator")
        logger.debug("Starting new simulation.")
        self.attack_graph.reset()
        self.attacker = Attacker(self.attack_graph, ['internet.connect'])
        return self._next_observation()

    def _next_observation(self):
        # Imperfect observations by intrusion detection system
        return np.array([self.attacker.observe(a) for a in self.attack_graph.attack_steps])

    def render(self, mode='human'):
        pass

    def disable(self, service):
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
                print("Defender disabling " + service)
                enabled_services[service] = 0
                enabled_services_status_changed = True
        obs, reward, done, info = env.step(tuple(enabled_services.values()))
        if info["time_on_current_step"] == 1 or enabled_services_status_changed:
            print(str(info['time']) + ": reward=" + str(reward) + ". Attacking " + str(info['current_step']))
