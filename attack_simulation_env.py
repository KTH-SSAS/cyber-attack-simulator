import gym
from gym import spaces
import numpy as np
import random

class AttackStep:

    def __init__(self, step_type='or', ttc=1, reward=0, children={}, isolator=None, true_positive=0.95, false_positive=0.1):
        self.step_type = step_type
        self.ttc = int(np.random.exponential(scale=ttc))
        self.reward = int(np.random.exponential(scale=reward))
        self.children = children
        self.parents = set()
        self.true_positive = true_positive
        self.false_positive = false_positive
        self.isolator = isolator


class AttackGraph:

    def __init__(self):
        self.reset()

    def reset(self):

        self.online = dict()
        self.online['lazarus.ftp'] = True
        self.online['lazarus.tomcat'] = True
        self.online['energetic_bear.apache'] = True
        self.online['sea_turtle.telnet'] = True
        self.online['buckeye.firefox'] = True
        
        self.attack_steps = {}
        self.attack_steps['lazarus.flag_adcb1f.capture'] = AttackStep(reward=1000, isolator='lazarus.ftp')
        self.attack_steps['lazarus.ftp.login'] = AttackStep(isolator='lazarus.ftp', children={'lazarus.flag_adcb1f.capture'})
        self.attack_steps['lazarus.ftp.dictionary_attack'] = AttackStep(ttc=100, isolator='lazarus.ftp', children={'lazarus.ftp.login'})
        self.attack_steps['lazarus.ftp.identify'] = AttackStep(isolator='lazarus.ftp', children={'lazarus.ftp.dictionary_attack'})

        self.attack_steps['buckeye.flag_14ce18.capture'] = AttackStep(reward=1000, isolator='buckeye.firefox')
        self.attack_steps['buckeye.firefox.connect'] = AttackStep(isolator='buckeye.firefox')

        self.attack_steps['energetic_bear.flag_73cb43.capture'] = AttackStep(reward=1000, isolator='energetic_bear.apache')
        self.attack_steps['energetic_bear.flag_3b2000.capture'] = AttackStep(reward=1000, isolator='energetic_bear.apache')
        self.attack_steps['energetic_bear.flag_de3b1c.capture'] = AttackStep(reward=1000, isolator='energetic_bear.apache')
        self.attack_steps['energetic_bear.flag_521bce.capture'] = AttackStep(reward=1000, isolator='energetic_bear.apache')
        self.attack_steps['energetic_bear.capture_traffic'] = AttackStep(ttc=5, isolator='energetic_bear.apache', children={'buckeye.firefox.connect', 'buckeye.flag_14ce18.capture'})
        self.attack_steps['energetic_bear.escalate_to_root'] = AttackStep(ttc=20, isolator='energetic_bear.apache', children={'energetic_bear.capture_traffic', 'energetic_bear.flag_73cb43.capture'})
        self.attack_steps['energetic_bear.pop_shell'] = AttackStep(ttc=10, isolator='energetic_bear.apache', children={'energetic_bear.escalate_to_root', 'energetic_bear.flag_3b2000.capture'})
        self.attack_steps['energetic_bear.apache.exploit_vulnerability'] = AttackStep(ttc=20, isolator='energetic_bear.apache', children={'energetic_bear.pop_shell', 'energetic_bear.flag_de3b1c.capture'})
        self.attack_steps['energetic_bear.apache.find_vulnerability'] = AttackStep(ttc=30, isolator='energetic_bear.apache', children={'energetic_bear.apache.exploit_vulnerability'})
        self.attack_steps['energetic_bear.apache.crawl'] = AttackStep(ttc=10, isolator='energetic_bear.apache', children={'energetic_bear.apache.find_vulnerability'})
        self.attack_steps['energetic_bear.apache.identify'] = AttackStep(isolator='energetic_bear.apache', children={'energetic_bear.apache.crawl', 'energetic_bear.flag_521bce.capture'})

        self.attack_steps['sea_turle.flag_6be6ef.capture'] = AttackStep(reward=1000, isolator='sea_turtle.telnet')
        self.attack_steps['sea_turle.flag_f9038f.capture'] = AttackStep(reward=1000, isolator='sea_turtle.telnet')
        self.attack_steps['sea_turtle.escalate_to_root'] = AttackStep(ttc=50, isolator='sea_turtle.telnet', children={'sea_turle.flag_6be6ef.capture'})
        self.attack_steps['sea_turtle.telnet.login'] = AttackStep(step_type='and', ttc=100, isolator='sea_turtle.telnet', children={'sea_turtle.escalate_to_root', 'sea_turle.flag_f9038f.capture'})
        self.attack_steps['sea_turtle.telnet.obtain_credentials'] = AttackStep(isolator='lazarus.tomcat', children={'sea_turtle.telnet.login'})
        self.attack_steps['sea_turtle.telnet.identify'] = AttackStep(isolator='sea_turtle.telnet', children={'sea_turtle.telnet.login'})

        self.attack_steps['lazarus.flag_90b353.capture'] = AttackStep(reward=1000, isolator='lazarus.tomcat')
        self.attack_steps['lazarus.flag_cd699a.capture'] = AttackStep(reward=1000, isolator='lazarus.tomcat')
        self.attack_steps['lazarus.dump_hashes'] = AttackStep(ttc=10, isolator='lazarus.tomcat', children={'sea_turtle.telnet.obtain_credentials'})
        self.attack_steps['lazarus.pop_shell'] = AttackStep(isolator='lazarus.tomcat', children={'lazarus.dump_hashes', 'lazarus.flag_cd699a.capture'})
        self.attack_steps['lazarus.tomcat.exploit_vulnerability'] = AttackStep(ttc=10, isolator='lazarus.tomcat', children={'lazarus.pop_shell'})
        self.attack_steps['lazarus.tomcat.find_vulnerability'] = AttackStep(ttc=10, isolator='lazarus.tomcat', children={'lazarus.tomcat.exploit_vulnerability'})
        self.attack_steps['lazarus.tomcat.dictionary_attack'] = AttackStep(ttc=10, isolator='lazarus.tomcat', children={'lazarus.tomcat.find_vulnerability', 'lazarus.flag_90b353.capture'})
        self.attack_steps['lazarus.tomcat.crawl'] = AttackStep(ttc=5, isolator='lazarus.tomcat', children={'lazarus.tomcat.dictionary_attack'})
        self.attack_steps['lazarus.tomcat.identify'] = AttackStep(isolator='lazarus.tomcat', children={'lazarus.tomcat.crawl'})

        self.attack_steps['office_network.map'] = AttackStep(ttc=10, children={'lazarus.ftp.identify', 'energetic_bear.apache.identify', 'lazarus.tomcat.identify', 'sea_turtle.telnet.identify'})
        self.attack_steps['internet.connect'] = AttackStep(children={'office_network.map'})
        
        self.record_parents()
        self.size = len(self.attack_steps)

    def record_parents(self):
        for parent in self.attack_steps:
            for child in self.attack_steps[parent].children:
                self.attack_steps[child].parents.add(parent)

    def steps_secured_by_isolating(self, service):
        steps = [asn for asn in self.attack_steps.keys() if self.attack_steps[asn].isolator == service]
        return steps

    def isolate(self, service):
        if service == 'lazarus.ftp':
            self.attack_steps['office_network.map'].children -= {'lazarus.ftp.identify'}
        if service == 'lazarus.tomcat':
            self.attack_steps['office_network.map'].children -= {'lazarus.tomcat.identify'}
        if service == 'energetic_bear.apache':
            self.attack_steps['office_network.map'].children -= {'energetic_bear.apache.identify'}
        if service == 'sea_turtle.telnet':
            self.attack_steps['office_network.map'].children -= {'sea_turtle.telnet.identify'}
            self.attack_steps['sea_turtle.telnet.obtain_credentials'].children -= {'sea_turtle.telnet.login'}

class Attacker:
    
    def __init__(self, attack_graph, compromised_steps):
        self.attack_graph = attack_graph
        self.compromised_steps = compromised_steps
        self.choose_next_step()
        self.time_on_current_step = 0
        self.total_time = 0

    def get_step(self, name):
        return self.attack_graph.attack_steps[name]
    
    def attack_surface(self, debug=False):
        # The attack surface consists of all reachable but uncompromised attack steps.
        att_surf = set()
        for compromised_step_name in self.compromised_steps:
            for child_name in self.get_step(compromised_step_name).children:
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

        att_surf -= self.compromised_steps
        return att_surf

    def choose_next_step(self): 
        # The attacker strategy is to select a random attack step of the available ones (i.e. from the attack surface).
        self.current_step = None
        if self.attack_surface():
            self.current_step = random.choice(list(self.attack_surface()))

    def attack(self):
        # If the attacker has run out of attack steps, then terminate.
        if not self.current_step:
            return False
        # If the attacker has spent the required time on the current attack step, then it becomes compromised.
        if self.time_on_current_step >= self.get_step(self.current_step).ttc:
            self.compromised_steps.add(self.current_step)
            # If the attack surface (the available uncompromised attack steps) is empty, then terminate.
            if not self.attack_surface():
                return False
            self.choose_next_step()
            self.time_on_current_step = 0
        # Keep track of the time spent.
        self.time_on_current_step += 1
        self.total_time += 1
        return True

    def reward(self):
        r = 0
        for cs in self.compromised_steps:
            r += self.get_step(cs).reward
        return r

    def observe(self, attack_step):
        rnd = random.uniform(0,1)
        if attack_step in self.compromised_steps:
            if rnd <= self.get_step(attack_step).true_positive:
                return True
            else:
                return False
        else:
            if rnd <= self.get_step(attack_step).false_positive:
                return True
            else:
                return False


class AttackSimulationEnv(gym.Env):

    def __init__(self):
        super(AttackSimulationEnv, self).__init__()
        self.attack_graph = AttackGraph()
        self.attacker = Attacker(self.attack_graph, {'internet.connect'})
        self.provision_reward = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.attack_graph.size, 1), dtype=np.float32)
        self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2)))

    def get_info(self):
        if self.attacker.current_step:
            info = {"time": self.attacker.total_time, "current_step": self.attacker.current_step, "time_on_current_step": self.attacker.time_on_current_step, "ttc_of_current_step": self.attacker.get_step(env.attacker.current_step).ttc, "attack_surface": self.attacker.attack_surface(), "self.attack_graph.online": self.attack_graph.online}
        else:
            info = {"time": self.attacker.total_time, "current_step": None, "time_on_current_step": None, "ttc_of_current_step": None, "attack_surface": self.attacker.attack_surface(), "self.attack_graph.online": self.attack_graph.online}
        return info

    def step(self, action):
        # The order of actions follows self.attack_graph.online
        action_id = 0
        # Isolate services according to the actions provided
        for service in self.attack_graph.online:
            if self.attack_graph.online[service]:
                self.provision_reward += 1
                if action[action_id] == 0:
                    self.isolate(service)
            action_id += 1

        obs = self._next_observation()
        # Positive rewards for maintaining services online and negative for compromised flags.
        reward = self.provision_reward - self.attacker.reward()
        
        # The attacker attacks. If the attacker's attack surface is empty, then the game ends.
        attacker_done = not self.attacker.attack()

        return obs, reward, attacker_done, self.get_info()

    def reset(self):
        self.attack_graph.reset()
        return self._next_observation()

    def _next_observation(self):
        return np.array([self.attacker.observe(a) for a in self.attack_graph.attack_steps])

    def render(self, mode='human'):
        pass

    def isolate(self, service):
        self.attack_graph.online[service] = False
        self.attacker.compromised_steps -= set(self.attack_graph.steps_secured_by_isolating(service))
        self.attack_graph.isolate(service)
        self.attacker.choose_next_step()
    

env = AttackSimulationEnv()
obs = env.reset()
online = dict()
# Defender can act by isolating various services (found in env.attack_graph.online)
for service in env.attack_graph.online:
    online[service] = 1
done = False
# The probability that the defender will isolate a given service at a given step is given by ISOLATION_PROBABILITY.
ISOLATION_PROBABILITY = 0.005
while not done:
    online_status_changed = False
    for service in online:
        if online[service] == 1 and random.uniform(0,1) < ISOLATION_PROBABILITY:
            online[service] = 0
            online_status_changed = True
    obs, reward, done, info = env.step(tuple(online.values()))
    assert env.attack_graph.attack_steps['office_network.map'].children == env.attacker.attack_graph.attack_steps['office_network.map'].children
    if info["time_on_current_step"] == 1 or online_status_changed:
        print("info: " + str(info) + " reward: " + str(reward))
print("Final: info: " + str(info) + " reward: " + str(reward))
print("Compromised attack steps: " + str(env.attacker.compromised_steps))
