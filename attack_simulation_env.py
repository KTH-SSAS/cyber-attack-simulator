import gym
from gym import spaces
import numpy as np
import random

class AttackStep:

    def __init__(self, step_type='or', ttc=1, reward=0, children={}, true_positive=0.95, false_positive=0.1):
        self.step_type = step_type
        self.ttc = int(np.random.exponential(scale=ttc))
        self.reward = int(np.random.exponential(scale=reward))
        self.children = children
        self.true_positive = true_positive
        self.false_positive = false_positive


class AttackGraph:

    def __init__(self):
        self.reset()

    def reset(self):
        self.attack_steps = {}
        self.attack_steps['capture_ftp_flag'] = AttackStep(reward=1000)
        self.attack_steps['ftp_login'] = AttackStep(children={'capture_ftp_flag'})
        self.attack_steps['dictionary_attack'] = AttackStep(ttc=100, children={'ftp_login'})
        self.attack_steps['identify_ftp_server'] = AttackStep(children={'dictionary_attack'})

        self.attack_steps['capture_root_flag'] = AttackStep(reward=2000)
        self.attack_steps['capture_user_flag'] = AttackStep(reward=500)
        self.attack_steps['capture_db_flag'] = AttackStep(reward=250)
        self.attack_steps['escalate_to_root'] = AttackStep(ttc=20, children={'capture_root_flag'})
        self.attack_steps['pop_shell'] = AttackStep(ttc=10, children={'escalate_to_root', 'capture_user_flag'})
        self.attack_steps['exploit_sqli'] = AttackStep(ttc=20, children={'pop_shell', 'capture_db_flag'})
        self.attack_steps['find_sqli'] = AttackStep(ttc=30, children={'exploit_sqli'})
        self.attack_steps['crawl_http_server'] = AttackStep(ttc=10, children={'find_sqli'})
        self.attack_steps['identify_http_server'] = AttackStep(children={'crawl_http_server'})

        self.attack_steps['map_network'] = AttackStep(ttc=10, children={'identify_ftp_server', 'identify_http_server'})
        self.attack_steps['internet'] = AttackStep(children={'map_network'})
        self.size = len(self.attack_steps)


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
        self.attacker = Attacker({'internet'})
        self.ftp_is_online = True
        self.http_is_online = True
        self.provision_reward = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.attacker.attack_graph.size, 1), dtype=np.float32)
        self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2)))

    def get_info(self):
        if self.attacker.current_step:
            info = {"time": self.attacker.total_time, "current_step": self.attacker.current_step, "time_on_current_step": self.attacker.time_on_current_step, "ttc_of_current_step": self.attacker.get_step(env.attacker.current_step).ttc, "attack_surface": self.attacker.attack_surface(), "ftp_is_online": self.ftp_is_online, "http_is_online": self.http_is_online}
        else:
            info = {"time": self.attacker.total_time, "current_step": None, "time_on_current_step": None, "ttc_of_current_step": None, "attack_surface": self.attacker.attack_surface(), "ftp_is_online": self.ftp_is_online, "http_is_online": self.http_is_online}
        return info

    def step(self, action):
        if self.ftp_is_online:
            self.provision_reward += 1
            if action[0] == 0:
                self.isolate_ftp()
        if self.http_is_online:
            self.provision_reward += 1
            if action[1] == 0:
                self.isolate_http()

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

    def isolate_ftp(self):
        self.ftp_is_online = False
        self.attacker.compromised_steps -= {'identify_ftp_server', 'dictionary_attack', 'ftp_login', 'capture_ftp_flag'}
        self.attacker.get_step('map_network').children -= {'identify_ftp_server'}
        self.attacker.choose_next_step()

    def isolate_http(self):
        self.http_is_online = False
        self.attacker.compromised_steps -= {'capture_root_flag', 'capture_user_flag', 'capture_db_flag', 'escalate_to_root', 'pop_shell', 'exploit_sqli', 'find_sqli', 'crawl_http_server', 'identify_http_server'}
        self.attacker.get_step('map_network').children -= {'identify_http_server'}
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
