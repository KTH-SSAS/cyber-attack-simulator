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
        
        self.attack_steps = {}
        self.attack_steps['lazarus.flag_adcb1f.capture'] = AttackStep(reward=1000, isolator='lazarus.ftp')
        self.attack_steps['lazarus.ftp.login'] = AttackStep(isolator='lazarus.ftp', children={'lazarus.flag_adcb1f.capture'})
        self.attack_steps['lazarus.ftp.dictionary_attack'] = AttackStep(ttc=100, isolator='lazarus.ftp', children={'lazarus.ftp.login'})
        self.attack_steps['lazarus.ftp.identify'] = AttackStep(isolator='lazarus.ftp', children={'lazarus.ftp.dictionary_attack'})

        self.attack_steps['energetic_bear.flag_73cb43.capture'] = AttackStep(reward=1000, isolator='energetic_bear.apache')
        self.attack_steps['energetic_bear.flag_3b2000.capture'] = AttackStep(reward=1000, isolator='energetic_bear.apache')
        self.attack_steps['energetic_bear.flag_de3b1c.capture'] = AttackStep(reward=1000, isolator='energetic_bear.apache')
        self.attack_steps['energetic_bear.flag_521bce.capture'] = AttackStep(reward=1000, isolator='energetic_bear.apache')
        self.attack_steps['energetic_bear.apache.escalate_to_root'] = AttackStep(ttc=20, isolator='energetic_bear.apache', children={'energetic_bear.flag_73cb43.capture'})
        self.attack_steps['energetic_bear.apache.pop_shell'] = AttackStep(ttc=10, isolator='energetic_bear.apache', children={'energetic_bear.apache.escalate_to_root', 'energetic_bear.flag_3b2000.capture'})
        self.attack_steps['energetic_bear.apache.exploit_sqli'] = AttackStep(ttc=20, isolator='energetic_bear.apache', children={'energetic_bear.apache.pop_shell', 'energetic_bear.flag_de3b1c.capture'})
        self.attack_steps['energetic_bear.apache.find_sqli'] = AttackStep(ttc=30, isolator='energetic_bear.apache', children={'energetic_bear.apache.exploit_sqli'})
        self.attack_steps['energetic_bear.apache.crawl'] = AttackStep(ttc=10, isolator='energetic_bear.apache', children={'energetic_bear.apache.find_sqli'})
        self.attack_steps['energetic_bear.apache.identify'] = AttackStep(isolator='energetic_bear.apache', children={'energetic_bear.apache.crawl', 'energetic_bear.flag_521bce.capture'})

        self.attack_steps['sea_turle.flag_f9038f.capture'] = AttackStep(reward=1000, isolator='sea_turtle.telnet')
        self.attack_steps['sea_turtle.telnet.login'] = AttackStep(step_type='and', ttc=100, isolator='sea_turtle.telnet', children={'sea_turle.flag_f9038f.capture'})
        self.attack_steps['sea_turtle.telnet.identify'] = AttackStep(isolator='sea_turtle.telnet', children={'sea_turtle.telnet.login'})

        self.attack_steps['lazarus.flag_90b353.capture'] = AttackStep(reward=1000, isolator='lazarus.tomcat')
        self.attack_steps['lazarus.flag_cd699a.capture'] = AttackStep(reward=1000, isolator='lazarus.tomcat')
        self.attack_steps['lazarus.tomcat.crack_hashes'] = AttackStep(ttc=100, isolator='lazarus.tomcat', children={'sea_turtle.telnet.login'})
        self.attack_steps['lazarus.tomcat.dump_hashes'] = AttackStep(isolator='lazarus.tomcat', children={'lazarus.tomcat.crack_hashes'})
        self.attack_steps['lazarus.tomcat.pop_shell'] = AttackStep(isolator='lazarus.tomcat', children={'lazarus.tomcat_dump_hashes', 'lazarus.flag_cd699a.capture'})
        self.attack_steps['lazarus.tomcat.upload_war'] = AttackStep(ttc=10, isolator='lazarus.tomcat', children={'lazarus.tomcat.pop_shell'})
        self.attack_steps['lazarus.tomcat.dictionary_attack'] = AttackStep(ttc=10, isolator='lazarus.tomcat', children={'lazarus.flag_90b353.capture'})
        self.attack_steps['lazarus.tomcat.crawl'] = AttackStep(ttc=5, isolator='lazarus.tomcat', children={'lazarus.tomcat.dictionary_attack'})
        self.attack_steps['lazarus.tomcat.identify'] = AttackStep(isolator='lazarus.tomcat', children={'lazarus.tomcat.crawl'})

        self.attack_steps['office_network.map'] = AttackStep(ttc=10, children={'lazarus.ftp.identify', 'energetic_bear.apache.identify', 'lazarus.tomcat.identify'})
        self.attack_steps['internet.connect'] = AttackStep(children={'office_network.map'})
        self.size = len(self.attack_steps)

    def steps_secured_by_isolating(self, service):
        steps = [asn for asn in self.attack_steps.keys() if self.attack_steps[asn].isolator == service]
        return steps

    def isolate(self, service):
        if service == 'lazarus.ftp':
            self.attack_steps['office_network.map'].children -= {'lazarus.ftp.identify'}
        if service == 'energetic_bear.apache':
            self.attack_steps['office_network.map'].children -= {'energetic_bear.apache.identify'}

class Attacker:
    
    def __init__(self, compromised_steps):
        self.attack_graph = AttackGraph()
        self.compromised_steps = compromised_steps
        self.choose_next_step()
        self.time_on_current_step = 0
        self.total_time = 0

    def get_step(self, name):
        return self.attack_graph.attack_steps[name]
    
    def attack_surface(self, debug=False):
         return set([child for step_name in self.compromised_steps for child in self.get_step(step_name).children]) - self.compromised_steps

    def choose_next_step(self): 
        self.current_step = None
        if self.attack_surface():
            self.current_step = random.choice(list(self.attack_surface()))

    def attack(self):
        if not self.current_step:
            return False
        if self.time_on_current_step >= self.get_step(self.current_step).ttc:
            self.compromised_steps.add(self.current_step)
            if not self.attack_surface():
                return False
            self.choose_next_step()
            self.time_on_current_step = 0
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
        self.attacker = Attacker({'internet.connect'})
        self.provision_reward = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.attacker.attack_graph.size, 1), dtype=np.float32)
        self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2)))

    def get_info(self):
        if self.attacker.current_step:
            info = {"time": self.attacker.total_time, "current_step": self.attacker.current_step, "time_on_current_step": self.attacker.time_on_current_step, "ttc_of_current_step": self.attacker.get_step(env.attacker.current_step).ttc, "attack_surface": self.attacker.attack_surface(), "self.attacker.attack_graph.online": self.attacker.attack_graph.online}
        else:
            info = {"time": self.attacker.total_time, "current_step": None, "time_on_current_step": None, "ttc_of_current_step": None, "attack_surface": self.attacker.attack_surface(), "self.attacker.attack_graph.online": self.attacker.attack_graph.online}
        return info

    def step(self, action):
        if self.attacker.attack_graph.online['lazarus.ftp']:
            self.provision_reward += 1
            if action[0] == 0:
                self.isolate('lazarus.ftp')
        if self.attacker.attack_graph.online['energetic_bear.apache']:
            self.provision_reward += 1
            if action[1] == 0:
                self.isolate('energetic_bear.apache')

        obs = self._next_observation()
        reward = self.provision_reward - self.attacker.reward()
        
        attacker_done = not self.attacker.attack()
        defender_done = action[0] + action[1] == 0        

        return obs, reward, attacker_done or defender_done, self.get_info()

    def reset(self):
        self.attacker.attack_graph.reset()
        return self._next_observation()

    def _next_observation(self):
        return np.array([self.attacker.observe(a) for a in self.attacker.attack_graph.attack_steps])

    def render(self, mode='human'):
        pass

    def isolate(self, service):
        self.attacker.attack_graph.online[service] = False
        self.attacker.compromised_steps -= set(self.attacker.attack_graph.steps_secured_by_isolating(service))
        self.attacker.attack_graph.isolate(service)
        self.attacker.choose_next_step()
    

env = AttackSimulationEnv()
obs = env.reset()
ftp_online = 1
http_online = 1
done = False
while not done:
    online_status_changed = False
    if ftp_online == 1 and random.uniform(0,1) > 0.99:
        ftp_online = 0
        online_status_changed = True
    if http_online == 1 and random.uniform(0,1) > 0.99:
        http_online = 0
        online_status_changed = True
    obs, reward, done, info = env.step((ftp_online, http_online))
    if info["time_on_current_step"] == 1 or online_status_changed:
        print("info: " + str(info) + " reward: " + str(reward))
print("Final: info: " + str(info) + " obs: " + str(obs) + " reward: " + str(reward))
