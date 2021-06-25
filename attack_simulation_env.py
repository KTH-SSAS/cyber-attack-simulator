import gym
from gym import spaces
import numpy as np
import random

class AttackStep:

    def __init__(self, step_type='or', ttc=1, reward=0, children={}, true_positive=1, false_positive=0):
        self.step_type = step_type
        self.ttc = ttc
        self.reward = reward
        self.children = children
        self.true_positive = true_positive
        self.false_positive = false_positive
        


class AttackGraph:

    def __init__(self):
        self.attack_steps = {}
        self.attack_steps['capture_ftp_flag'] = AttackStep(reward=1000)
        self.attack_steps['ftp_login'] = AttackStep(children={'capture_ftp_flag'})
        self.attack_steps['dictionary_attack'] = AttackStep(ttc=100, children={'ftp_login'})
        self.attack_steps['identify_ftp'] = AttackStep(children={'dictionary_attack'})
        self.attack_steps['map_network'] = AttackStep(ttc=10, children={'identify_ftp'})
        self.attack_steps['internet'] = AttackStep(children={'map_network'})
        self.size = len(self.attack_steps)



class Attacker:
    
    def __init__(self, attack_graph, compromised_steps):
        self.attack_graph = attack_graph
        self.compromised_steps = compromised_steps
        self.current_step = random.choice(list(self.attack_surface()))
        self.time_on_current_step = 0
        self.total_time = 0

    def attack_surface(self):
        return set([child for step_name in self.compromised_steps for child in self.attack_graph.attack_steps[step_name].children]) - self.compromised_steps

    def attack(self):
        if self.time_on_current_step >= self.attack_graph.attack_steps[self.current_step].ttc:
            self.compromised_steps.add(self.current_step)
            if not self.attack_surface():
                return False
            self.current_step = random.choice(list(self.attack_surface()))
            self.time_on_current_step = 0
        self.time_on_current_step += 1
        self.total_time += 1
        return True

    def reward(self):
        r = 0
        for cs in self.compromised_steps:
            r += self.attack_graph.attack_steps[cs].reward
        return r

    def observe(self, attack_step):
        rnd = random.uniform(0,1)
        if attack_step in self.compromised_steps:
            if rnd <= self.attack_graph.attack_steps[attack_step].true_positive:
                return True
            else:
                return False
        else:
            if rnd <= self.attack_graph.attack_steps[attack_step].false_positive:
                return True
            else:
                return False


class AttackSimulationEnv(gym.Env):

    def __init__(self):

        super(AttackSimulationEnv, self).__init__()

        self.attack_graph = AttackGraph()
        self.attacker = Attacker(self.attack_graph, {'internet'})
        self.ftp_is_online = True
        self.http_is_online = True
        self.provision_reward = 0


        self.observation_space = spaces.Box(low=0, high=1, shape=(self.attack_graph.size, 1), dtype=np.float32)
        self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2)))

    def step(self, action):
        if action[0] == 0 and self.ftp_is_online:
            self.isolate_ftp()
        if action[1] == 0 and self.http_is_online:
            self.isolate_http()

        if not self.ftp_is_online:
            self.provision_reward -= 1
        if not self.http_is_online:
            self.provision_reward -= 1

        attacker_done = not self.attacker.attack()
        defender_done = action[0] + action[1] == 0        

        obs = self._next_observation()

        reward = self.provision_reward - self.attacker.reward()

        info = {"time": self.attacker.total_time, "current_step": self.attacker.current_step, "time_on_current_step": self.attacker.time_on_current_step, "ttc_of_current_step": self.attacker.attack_graph.attack_steps[env.attacker.current_step].ttc}
        
        return obs, reward, attacker_done or defender_done, info

    def reset(self):
        self.attack_graph = AttackGraph()
        return self._next_observation()

    def _next_observation(self):
        return np.array([self.attacker.observe(a) for a in self.attack_graph.attack_steps])

    def render(self, mode='human'):
        pass

    def isolate_ftp(self):
        self.ftp_is_online = False
        self.attacker.compromised_steps -= {'identify_ftp', 'dictionary_attack', 'ftp_login', 'capture_ftp_flag', 'map_network'}
        self.attacker.current_step = "map_network"
        print("Isolating ftp")

    def isolate_http(self):
        self.http_is_online = False
        print("Isolating http")

    def reinstate_ftp(self):
        self.ftp_is_online = True

    def reinstate_http(self):
        self.http_is_online = True


env = AttackSimulationEnv()
obs = env.reset()
ftp_online = 1
http_online = 1
done = False
while not done:
    if ftp_online == 1 and random.uniform(0,1) > 0.99:
        ftp_online = 0
    if http_online == 1 and random.uniform(0,1) > 0.99:
        http_online = 0
    obs, reward, done, info = env.step((ftp_online, http_online))
    if info["time_on_current_step"] == 1:
        print("obs: " + str(obs) + " reward: " + str(reward) + " info: " + str(info))
