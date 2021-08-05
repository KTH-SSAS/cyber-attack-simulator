import logging
import random
from typing import Dict, Set

import gym
import numpy as np
from gym import spaces

from .graph import AttackGraph

logger = logging.getLogger("simulator")


class Attacker:
    def __init__(
        self,
        attack_graph: AttackGraph,
        ttc: Dict[str, float],
        rewards: Dict[str, float],
        true_positive: float,
        false_positive: float,
        compromised_steps: Set[str],
        deterministic=False,
        strategy="random",
    ):
        self.strategy = strategy
        self.g = attack_graph
        # self.compromised_steps keeps track of attack steps reached by that attacker.
        self.compromised_steps = compromised_steps
        self.deterministic = deterministic
        self.time_on_current_step = 0
        self.total_time = 0
        self.current_step = None
        self.ttc = ttc
        self.rewards = rewards
        self.true_positive = true_positive
        self.false_positive = false_positive
        self.enabled = set(self.g.attack_names)
        self.inefficiency_counter = 0
        self.choose_next_step()
        self.reward = 0

    @property
    def attack_surface(self):
        """The attack surface consists of all reachable but uncompromised attack steps."""
        self.inefficiency_counter += 1
        logger.debug(f"Recomputing attack surface ({self.inefficiency_counter})")
        att_surf = set()
        for compromised_step_name in self.compromised_steps:
            for child_name in self.g.attack_steps[compromised_step_name].children:
                if child_name in self.enabled:
                    if self.g.attack_steps[child_name].step_type == "or":
                        att_surf.add(child_name)
                    else:
                        all_parents_are_compromised = True
                        for parent_name in self.g.attack_steps[child_name].parents:
                            if parent_name not in self.compromised_steps:
                                all_parents_are_compromised = False
                                break
                        if all_parents_are_compromised:
                            att_surf.add(child_name)

        att_surf -= self.compromised_steps
        return att_surf

    def choose_next_step(self):
        if self.strategy == "random":
            step = self.choose_next_step_randomly()
        elif self.strategy == "value_maximizing":
            step = self.choose_highest_value_step()
        self.current_step = step

    def choose_next_step_randomly(self):
        """
        The attacker strategy is currently simply to select a random
        attack step of the available ones (i.e. from the attack surface).
        """
        if self.attack_surface:
            sorted_surface = sorted(self.attack_surface)
            if self.deterministic:
                step = sorted_surface[0]
            else:
                step = random.choice(sorted_surface)
        else:
            step = None
        return step

    def choose_highest_value_step(self):
        """
        Selecting the attack step with the highet net present value.
        Because the attacker cannot know when the defender might disable a service,
        future rewards are uncertain, and thus the introduction of the discount
        rate of the net present value calculation.
        Note: Does not consider AND steps, so will not always act optimally.
        """
        highest_value = 0
        step_value = dict()
        surface = self.attack_surface
        if surface:
            for step_name in surface:
                step_value[step_name] = self.value(step_name)
                if step_value[step_name] > highest_value:
                    highest_value = step_value[step_name]
                    step = step_name
        return step

    def value(self, parent_name, discount_rate=0.1):
        value = self.rewards[parent_name]
        for child_name in self.g.attack_steps[parent_name].children:
            value += self.value(child_name)
        value = value / (1 + discount_rate) ** self.ttc[parent_name]
        return value

    def attack(self, enabled):
        self.enabled = enabled
        self.inefficiency_counter = 0
        # If the attacker has run out of attack steps, then terminate.
        if not self.current_step:
            return False
        self.reward = 0
        # If the attacker has spent the required time on the current attack step,
        # then it becomes compromised.
        if self.time_on_current_step >= self.ttc[self.current_step]:
            self.compromised_steps.add(self.current_step)
            self.reward = self.rewards[self.current_step]
            # If the attack surface (the available uncompromised attack steps) is empty, terminate.
            compromised_now = self.current_step
            if not self.attack_surface:
                logger.debug(
                    "Step %.0f: Compromised %s. Nothing more to attack.",
                    self.total_time,
                    compromised_now,
                )
                return False
            self.choose_next_step()
            self.time_on_current_step = 0
            logger.debug(
                "Step %.0f: Compromised %s. Attacking %s.",
                self.total_time,
                compromised_now,
                self.current_step,
            )
        # Keep track of the time spent.
        self.time_on_current_step += 1
        self.total_time += 1
        return True

    def observe(self, attack_step):
        """
        Observations of the attacker are made by an intrusion detection system.
        The accuracy of observations is given for each attack step
        by the true and false positive rates respectively.
        """
        rnd = random.uniform(0, 1)
        if attack_step in self.compromised_steps:
            return rnd <= self.true_positive

        return rnd <= self.false_positive

    @property
    def compromised_flags(self):
        return [step for step in self.compromised_steps if "flag" in step]


class AttackSimulationEnv(gym.Env):
    def __init__(
        self,
        deterministic=False,
        early_flag_reward=1000,
        late_flag_reward=10000,
        final_flag_reward=100000,
        easy_ttc=10,
        hard_ttc=100,
        graph_size="large",
        attacker_strategy="random",
        true_positive=1.0,
        false_positive=0.0,
    ):
        super().__init__()
        self.deterministic = deterministic
        self.attack_graph = AttackGraph(
            dict(
                early_flag_reward=early_flag_reward,
                late_flag_reward=late_flag_reward,
                final_flag_reward=final_flag_reward,
                easy_ttc=easy_ttc,
                hard_ttc=hard_ttc,
                graph_size=graph_size,
            )
        )
        self.attacker_strategy = attacker_strategy
        self.true_positive = true_positive
        self.false_positive = false_positive

        # An observation informs the defender of which attack steps have been compromised.
        # Observations are imperfect.
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.attack_graph.num_attacks, 1), dtype=np.float32
        )
        # The defender action space consists of the disablement of services and hosts.
        self.n_defender_actions = self.attack_graph.num_services
        self.action_space = spaces.Tuple(([spaces.Discrete(2)] * self.n_defender_actions))
        self.prev_service = None
        self.step_number = 0

    def create_attacker(self):
        self.attacker = Attacker(
            self.attack_graph,
            self.ttc,
            self.rewards,
            self.true_positive,
            self.false_positive,
            {"internet.connect"},
            deterministic=self.deterministic,
            strategy=self.attacker_strategy,
        )

    def get_info(self):
        if self.attacker.current_step:
            info = {
                "time": self.attacker.total_time,
                "current_step": self.attacker.current_step,
                "time_on_current_step": self.attacker.time_on_current_step,
                "ttc_of_current_step": self.ttc[self.attacker.current_step],
                "attack_surface": self.attacker.attack_surface,
                "enabled_services": self.enabled_services,
            }
        else:
            info = {
                "time": self.attacker.total_time,
                "current_step": None,
                "time_on_current_step": None,
                "ttc_of_current_step": None,
                "attack_surface": self.attacker.attack_surface,
                "enabled_services": self.enabled_services,
            }
        return info

    def step(self, action):
        self.step_number += 1
        # provision_reward is the defender reward for maintaining services online.
        provision_reward = 0
        # Disable services according to the actions provided
        # The order of actions follows self.attack_graph.enabled_services
        for index, service in enumerate(self.attack_graph.service_names):
            if service in self.enabled_services:
                provision_reward += 1
                if action[index] == 0:
                    self._disable(service)

        # The attacker attacks. If the attacker's attack surface is empty, then the game ends.
        attacker_done = not self.attacker.attack(self.enabled_attacks)

        obs = self._next_observation()
        # Positive rewards for maintaining services and negative for compromised flags.
        reward = provision_reward - self.attacker.reward
        info = self.get_info()
        if attacker_done:
            logger.debug("Attacker is done.")
            logger.debug("Compromised steps: %s", str(self.attacker.compromised_steps))
            logger.debug("Compromised flags: %s", str(self.attacker.compromised_flags))
        info["compromised_steps"] = self.attacker.compromised_steps
        info["compromised_flags"] = self.attacker.compromised_flags
        return obs, reward, attacker_done, info

    def reset(self):
        logger.debug("Starting new simulation.")
        self.step_number = 0
        self.enabled_attacks = set(self.attack_graph.attack_names)
        self.enabled_services = set(self.attack_graph.service_names)
        self.ttc = {
            name: step.ttc
            if self.deterministic
            else 1
            if step.ttc <= 1
            else max(1, int(np.random.exponential(scale=step.ttc)))
            for name, step in self.attack_graph.attack_steps.items()
        }
        self.rewards = {
            name: step.reward
            if self.deterministic
            else int(np.random.exponential(scale=step.reward))
            for name, step in self.attack_graph.attack_steps.items()
        }
        self.create_attacker()
        return self._next_observation()

    def update_accuracy(self, true_positive, false_positive):
        # FIXME: not compliant with OpenAI Gym API
        self.true_positive = true_positive
        self.false_positive = false_positive

    def interpret_observation(self, observations):
        """Debug function"""
        # FIXME: not compliant with OpenAI Gym API
        compromised = []
        for i, obs in enumerate(observations):
            if obs:
                compromised.append(list(self.attack_graph.attack_steps)[i])
        return compromised

    def interpret_action_probabilities(self, action_probs):
        """Debug function"""
        # FIXME: not compliant with OpenAI Gym API
        act_prob_dict = {"no action": f"{action_probs[0]:.2f}"}
        for i in range(1, len(action_probs)):
            act_prob_dict[self.attack_graph.service_names[i - 1]] = f"{action_probs[i]:.2f}"
        return act_prob_dict

    def interpret_action(self, action):
        """Debug function"""
        # FIXME: not compliant with OpenAI Gym API
        # FIXME: action here appears to be an index, whereas action in step() is an array
        if action == 0:
            return "no action"

        return self.attack_graph.service_names[action - 1]

    def observation_from_compromised_steps(self, compromised_steps):
        """Debug function"""
        # FIXME: not compliant with OpenAI Gym API
        # FIXME: removed unused code
        return [a in compromised_steps for a in self.attack_graph.attack_names]

    def string_from_observation(self, observation):
        """Debug function"""
        # FIXME: not compliant with OpenAI Gym API
        return "".join([str(int(o)) for o in observation])

    def _next_observation(self):
        # Imperfect observations by intrusion detection system
        self.obs = np.array([self.attacker.observe(a) for a in self.attack_graph.attack_names])
        return self.obs

    def render(self, mode="human"):
        pass

    def _disable(self, service):
        if service != self.prev_service:
            if service in self.enabled_services:
                logger.debug(
                    "Step %d: Disabling %s because defender sees %s",
                    self.step_number,
                    service,
                    self.string_from_observation(self.obs),
                )
            else:
                logger.debug(
                    "Step %d: Re-disabling %s again, because defender sees %s",
                    self.step_number,
                    service,
                    self.string_from_observation(self.obs),
                )
        self.prev_service = service

        # Disable service
        self.enabled_services.remove(service)
        # Disconnect all attack steps that match the service.
        for attack_name in self.enabled_attacks.copy():
            if service in attack_name:
                self.enabled_attacks.remove(attack_name)
        # Also disable subservices (e.g. fancy_bear.ssh should be disabled when fancy_bear is)
        for subservice in self.enabled_services.copy():
            if service in subservice:
                self.enabled_services.remove(subservice)

        # The attacker might be attacking a step that just now became disabled,
        # so needs to choose a new one.
        self.attacker.choose_next_step()
