from typing import List
import random
import logging
from gym import spaces
import numpy as np
import gym
from attack_simulator.attack_graph import AttackGraph, AttackStep

# The probability that the defender will disable a given service at a given step is given by DISABLE_PROBABILITY.
DISABLE_PROBABILITY = 0.001
# For debugging convenience, simulations can be made deterministic, only dependent on the random seed.
DETERMINISTIC = False
RANDOM_SEED = 4


class Attacker:

    def __init__(self, attack_graph: AttackGraph, compromised_steps: List[str], deterministic=False, strategy='random'):
        self.strategy = strategy
        self.attack_graph = attack_graph
        # self.compromised_steps keeps track of which attack steps have been reached by that attacker.
        self.compromised_steps = compromised_steps
        self.deterministic = deterministic
        self.choose_next_step()
        self.time_on_current_step = 0
        self.total_time = 0
        self.reward = 0

    def get_step(self, name) -> AttackStep:
        return self.attack_graph.attack_steps[name]

    def attack_surface(self):
        """The attack surface consists of all reachable but uncompromised attack steps."""
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
        if self.strategy == 'random':
            self.choose_next_step_randomly()
        elif self.strategy == 'value_maximizing':
            self.choose_highest_value_step()

    def choose_next_step_randomly(self):
        """
        The attacker strategy is currently simply to select a random
        attack step of the available ones (i.e. from the attack surface).
        """
        self.current_step = None
        if self.attack_surface():
            if self.deterministic:
                sorted_surface = sorted(list(self.attack_surface()))
                self.current_step = sorted_surface[0]
            else:
                self.current_step = random.choice(list(self.attack_surface()))

    def choose_highest_value_step(self):
        """
        Selecting the attack step with the highet net present value.
        Because the attacker cannot know when the defender might disable a service,
        future rewards are uncertain, and thus the introduction of the discount
        rate of the net present value calculation.
        Note: Does not consider AND steps, so will not always act optimally.
        """
        self.current_step = None
        highest_value = 0
        step_value = dict()
        surface = self.attack_surface()
        if surface:
            for step_name in surface:
                step_value[step_name] = self.value(step_name)
                if step_value[step_name] > highest_value:
                    highest_value = step_value[step_name]
                    self.current_step = step_name

    def value(self, parent_name, discount_rate=0.1):
        parent = self.attack_graph.attack_steps[parent_name]
        value = parent.reward
        for child_name in parent.children:
            value += self.value(child_name)
        value = value/(1 + discount_rate)**parent.ttc
        return value

    def attack(self):
        logger = logging.getLogger("simulator")
        # If the attacker has run out of attack steps, then terminate.
        if not self.current_step:
            return False
        self.reward = 0
        # If the attacker has spent the required time on the current attack step, then it becomes compromised.
        if self.time_on_current_step >= self.get_step(self.current_step).ttc:
            self.compromised_steps.append(self.current_step)
            self.reward = self.attack_graph.attack_steps[self.current_step].reward
            # If the attack surface (the available uncompromised attack steps) is empty, then terminate.
            compromised_now = self.current_step
            if not self.attack_surface():
                logger.debug("Step %.0f: Compromised %s. Nothing more to attack.",
                             self.total_time, compromised_now)
                return False
            self.choose_next_step()
            self.time_on_current_step = 0
            logger.debug("Step %.0f: Compromised %s. Attacking %s.",
                         self.total_time, compromised_now, self.current_step)
        # Keep track of the time spent.
        self.time_on_current_step += 1
        self.total_time += 1
        return True

    def observe(self, attack_step):
        """
        Observations of the attacker are made by an intrusion detection system.
        The accuracy of observations is given for each attack step by the true and false positive rates respectively. 
        """
        rnd = random.uniform(0, 1)
        if attack_step in self.compromised_steps:
            return rnd <= self.get_step(attack_step).true_positive

        return rnd <= self.get_step(attack_step).false_positive

    @property
    def compromised_flags(self):
        return [step for step in self.compromised_steps if 'flag' in step]


class AttackSimulationEnv(gym.Env):

    def __init__(self, deterministic=False, early_flag_reward=1000, late_flag_reward=10000, final_flag_reward=100000, easy_ttc=10, hard_ttc=100, graph_size='large', attacker_strategy='random', true_positive=1.0, false_positive=0.0):
        super().__init__()
        self.deterministic = deterministic
        self.early_flag_reward = early_flag_reward
        self.late_flag_reward = late_flag_reward
        self.final_flag_reward = final_flag_reward
        self.easy_ttc = easy_ttc
        self.hard_ttc = hard_ttc
        self.true_positive = true_positive
        self.false_positive = false_positive
        self.attack_graph = AttackGraph(deterministic=deterministic, early_flag_reward=self.early_flag_reward,
                                        late_flag_reward=self.late_flag_reward, final_flag_reward=self.final_flag_reward, easy_ttc=self.easy_ttc, hard_ttc=self.hard_ttc, graph_size=graph_size, true_positive=self.true_positive, false_positive=self.false_positive)
        self.attacker_strategy = attacker_strategy
        self.create_attacker()
        # An observation informs the defender of which attack steps have been compromised.
        # Observations are imperfect.
        self.observation_space = spaces.Box(low=0, high=1, shape=(
            self.attack_graph.size, 1), dtype=np.float32)
        # The defender action space consists of the disablement of services and hosts.
        self.n_defender_actions = len(self.attack_graph.enabled_services)
        self.action_space = spaces.Tuple(
            ([spaces.Discrete(2)]*self.n_defender_actions))
        self.provision_reward = 0

    def create_attacker(self):
        self.attacker = Attacker(self.attack_graph, ['internet.connect'], deterministic=self.deterministic, strategy=self.attacker_strategy)

    def get_info(self):
        if self.attacker.current_step:
            info = {"time": self.attacker.total_time, "current_step": self.attacker.current_step, "time_on_current_step": self.attacker.time_on_current_step, "ttc_of_current_step": self.attacker.get_step(
                self.attacker.current_step).ttc, "attack_surface": self.attacker.attack_surface(), "self.attack_graph.enabled_services": self.attack_graph.enabled_services}
        else:
            info = {"time": self.attacker.total_time, "current_step": None, "time_on_current_step": None, "ttc_of_current_step": None,
                    "attack_surface": self.attacker.attack_surface(), "self.attack_graph.enabled_services": self.attack_graph.enabled_services}
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

        # The attacker attacks. If the attacker's attack surface is empty, then the game ends.
        attacker_done = not self.attacker.attack()

        obs = self._next_observation()
        # Positive rewards for maintaining services enabled_services and negative for compromised flags.
        reward = self.provision_reward - self.attacker.reward
        info = self.get_info()
        if attacker_done:
            logger.debug("Attacker is done.")
            logger.debug("Compromised steps: %s", str(
                self.attacker.compromised_steps))
            logger.debug("Compromised flags: %s", str(
                self.attacker.compromised_flags))
        info['compromised_steps'] = self.attacker.compromised_steps
        info['compromised_flags'] = self.attacker.compromised_flags
        return obs, reward, attacker_done, info

    def reset(self):
        logger = logging.getLogger("simulator")
        logger.debug("Starting new simulation.")
        self.attack_graph.reset()
        self.create_attacker()
        return self._next_observation()

    def interpret_observation(self, observations):
        """Debug function"""
        compromised = []
        for i, obs in enumerate(observations):
            if obs:
                compromised.append(list(self.attack_graph.attack_steps)[i])
        return compromised

    def interpret_action_probabilities(self, action_probs):
        """Debug function"""
        act_prob_dict = {"no action": f"{action_probs[0]:.2f}"}
        for i in range(1, len(action_probs)):
            act_prob_dict[list(self.attack_graph.enabled_services)[
                i-1]] = f"{action_probs[i]:.2f}"
        return act_prob_dict

    def interpret_action(self, action):
        """Debug function"""
        if action == 0:
            return "no action"

        return list(self.attack_graph.enabled_services)[action-1]

    def observation_from_compromised_steps(self, compromised_steps):
        """Debug function"""
        return np.array([a in compromised_steps for a in self.attack_graph.attack_steps])

    def _next_observation(self):
        # Imperfect observations by intrusion detection system
        return np.array([self.attacker.observe(a) for a in self.attack_graph.attack_steps])

    def render(self, mode='human'):
        pass

    def disable(self, service):
        logger = logging.getLogger('simulator')
        if self.attack_graph.enabled_services[service]:
            logger.debug("Disabling %s while attacker is attacking %s",
                         service, self.attacker.current_step)
        self.attack_graph.disable(service)
        #self.attacker.choose_next_step()
