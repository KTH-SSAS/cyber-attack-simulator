import logging
from dataclasses import asdict
from typing import List, Union

import gym
import gym.spaces as spaces
import numpy as np

from attack_simulator.agents import ATTACKERS
from attack_simulator.agents.agent import Agent
from attack_simulator.config import EnvConfig

from .graph import AttackGraph
from .renderer import AttackSimulationRenderer
from .rng import get_rng
from .utils import enabled

logger = logging.getLogger("simulator")


class AttackSimulationEnv(gym.Env):
    NO_ACTION = "no action"

    def __init__(self, config: Union[EnvConfig, dict]):
        super(AttackSimulationEnv, self).__init__()

        if isinstance(config, dict):
            config = EnvConfig(**config)

        # process configuration, leave the graph last, as it may destroy env_config
        self.config = config
        self.attacker_class = ATTACKERS[config.attacker]
        self.true_positive = config.true_positive
        self.false_positive = config.false_positive
        self.save_graphs = config.save_graphs
        self.save_logs = config.save_logs
        self.g: AttackGraph = AttackGraph(config.graph_config)

        # prepare just enough to get dimensions sorted, do the rest on first `reset`
        self.entry_attack_index = self.g.attack_names.index(self.g.root)

        # An observation informs the defender of
        # a) which services are turned on; and,
        # b) which attack steps have been successfully taken
        self.dim_observations = self.g.num_services + self.g.num_attacks
        self.observation_space = spaces.Tuple((spaces.Discrete(2),) * self.dim_observations)

        # The defender action space allows to disable any one service or leave all unchanged
        self.num_actions = self.g.num_services + 1
        self.action_space = spaces.Discrete(self.num_actions)

        self.episode_count = 0
        self._seed = None
        self.done = False
        self.reward = None
        self.action = 0
        self.attack_index = None
        self.compromised_flags: List[str] = []
        self.compromised_steps: List[str] = []
        self.renderer = None

        self.episode_id = self._get_episode_id()

    def _extract_attack_step_field(self, field_name):
        return np.array(
            [
                asdict(self.g.attack_steps[attack_name])[field_name]
                for attack_name in self.g.attack_names
            ]
        )

    def _get_episode_id(self):
        # TODO connect this with ray run id/wandb run id instead of random seed.
        return f"{self._seed}_{self.episode_count}"

    def _setup(self):
        # prime RNG if not yet set by `seed`
        if self._seed is None:
            self.seed()

        self.dependent_services = [
            [dependent.startswith(main) for dependent in self.g.service_names]
            for main in self.g.service_names
        ]

        self.attack_prerequisites = [
            (
                # required services
                [attack_name.startswith(service_name) for service_name in self.g.service_names],
                # logic function to combine prerequisites
                any if self.g.attack_steps[attack_name].step_type == "or" else all,
                # prerequisite attack steps
                [
                    prerequisite_name in self.g.attack_steps[attack_name].parents
                    for prerequisite_name in self.g.attack_names
                ],
            )
            for attack_name in self.g.attack_names
        ]

        self.ttc_params = self._extract_attack_step_field("ttc")
        self.reward_params = self._extract_attack_step_field("reward")

    def reset(self):
        if self.episode_count == 0:
            self._setup()

        self.episode_count += 1
        self.episode_id = self._get_episode_id()
        logger.debug(f"Starting new simulation. (#{self.episode_id})")

        self.ttc_remaining = np.array(
            [max(1, int(v)) for v in self.rng.exponential(self.ttc_params)]
        )
        self.rewards = np.array([int(v) for v in self.rng.exponential(self.reward_params)])

        self.simulation_time = 0
        self.service_state = np.full(self.g.num_services, 1)
        self.attack_state = np.full(self.g.num_attacks, 0)
        self.attack_surface = np.full(self.g.num_attacks, 0)
        self.attack_surface[self.entry_attack_index] = 1

        self.attacker: Agent = self.attacker_class(
            dict(
                attack_graph=self.g,
                ttc=self.ttc_remaining,
                rewards=self.rewards,
                random_seed=self._seed + self.episode_count,
            )
        )
        return self.observe()

    def observe(self):
        # Observation of attack steps is subject to the true/false positive rates
        # of an assumed underlying intrusion detection system
        # Depending on the true and false positive rates for each step,
        # ongoing attacks may not be reported, or non-existing attacks may be spuriously reported
        probabilities = self.rng.uniform(0, 1, self.g.num_attacks)
        true_positives = self.attack_state & (probabilities <= self.true_positive)
        false_positives = (1 - self.attack_state) & (probabilities <= self.false_positive)
        detected = true_positives | false_positives
        self.observation = tuple(np.append(self.service_state, detected))
        return self.observation

    def step(self, action):
        self.action = action
        assert 0 <= action < self.num_actions

        self.simulation_time += 1
        attacker_reward = 0

        # reserve 0 for no action
        if action:
            # decrement to obtain index
            service = action - 1
            # only disable services that are still on
            if self.service_state[service]:
                # disable the service itself and any dependent services
                self.service_state[self.dependent_services[service]] = 0
                # remove dependent attacks from the attack surface
                for attack_index in np.flatnonzero(self.attack_surface):
                    required_services, _, _ = self.attack_prerequisites[attack_index]
                    if not all(enabled(required_services, self.service_state)):
                        self.attack_surface[attack_index] = 0

                # end episode when attack surface becomes empty
                self.done = not any(self.attack_surface)

        self.attack_index = None

        if not self.done:
            # obtain attacker action, this _can_ be 0 for no action
            self.attack_index = self.attacker.act(self.attack_surface) - 1
            assert -1 <= self.attack_index < self.g.num_attacks

            if self.attack_index != -1:
                # compute attacker reward
                self.ttc_remaining[self.attack_index] -= 1
                if self.ttc_remaining[self.attack_index] == 0:
                    # successful attack, update reward, attack_state, attack_surface
                    attacker_reward = self.rewards[self.attack_index]
                    self.attack_state[self.attack_index] = 1
                    self.attack_surface[self.attack_index] = 0

                    # add eligible children to the attack surface
                    children = self.g.attack_steps[self.g.attack_names[self.attack_index]].children
                    for child_name in children:
                        child_index = self.g.attack_names.index(child_name)
                        required_services, logic, prerequisites = self.attack_prerequisites[
                            child_index
                        ]
                        if (
                            not self.attack_state[child_index]
                            and all(enabled(required_services, self.service_state))
                            and logic(enabled(prerequisites, self.attack_state))
                        ):
                            self.attack_surface[child_index] = 1

                    # end episode when attack surface becomes empty
                    self.done = not any(self.attack_surface)

            # TODO: placeholder, none of the current attackers learn...
            # self.attacker.update(attack_surface, attacker_reward, self.done)

        # compute defender reward
        # positive reward for maintaining services online (1 unit per service)
        # negative reward for the attacker's gains (as measured by `attacker_reward`)
        # FIXME: the reward for maintaining services is _very_ low
        self.reward = sum(self.service_state) - attacker_reward

        self.compromised_steps = self._interpret_attacks()
        self.compromised_flags = [
            step_name for step_name in self.compromised_steps if "flag" in step_name
        ]

        info = {
            "time": self.simulation_time,
            "attack_surface": self.attack_surface,
            "current_step": None
            if self.attack_index is None
            else self.NO_ACTION
            if self.attack_index == -1
            else self.g.attack_names[self.attack_index],
            "ttc_remaining_on_current_step": -1
            if self.attack_index is None or self.attack_index == -1
            else self.ttc_remaining[self.attack_index],
            "compromised_steps": self.compromised_steps,
            "compromised_flags": self.compromised_flags,
        }

        if self.done:
            logger.debug("Attacker done")
            logger.debug(f"Compromised steps: {self.compromised_steps}")
            logger.debug(f"Compromised flags: {self.compromised_flags}")

        return self.observe(), self.reward, self.done, info

    def _interpret_services(self, services=None):
        if services is None:
            services = self.service_state
        return list(np.array(self.g.service_names)[np.flatnonzero(services)])

    def _interpret_attacks(self, attacks=None):
        if attacks is None:
            attacks = self.attack_state
        return list(np.array(self.g.attack_names)[np.flatnonzero(attacks)])

    def _interpret_observation(self, observation=None):
        if observation is None:
            observation = self.observation
        services = observation[: self.g.num_services]
        attacks = observation[self.g.num_services :]
        return self._interpret_services(services), self._interpret_attacks(attacks)

    def _interpret_action(self, action):
        return (
            self.NO_ACTION
            if action == 0
            else self.g.service_names[action - 1]
            if 0 < action <= self.g.num_services
            else "invalid action"
        )

    def _interpret_action_probabilities(self, action_probabilities):
        keys = [self.NO_ACTION] + self.g.service_names
        return {key: value for key, value in zip(keys, action_probabilities)}

    def render(self, mode="human"):
        if not self.renderer:
            self.renderer = AttackSimulationRenderer(self)
        self.renderer.render()
        return True

    def seed(self, seed=None):
        self.rng, self._seed = get_rng(seed)
        return self._seed
