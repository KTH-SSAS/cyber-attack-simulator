import logging
from typing import List, Union

import gym
import gym.spaces as spaces
import numpy as np

from attack_simulator.agents import ATTACKERS
from attack_simulator.agents.agent import Agent
from attack_simulator.config import EnvConfig
from attack_simulator.rng import get_rng
from attack_simulator.sim import AttackSimulator

from .renderer import AttackSimulationRenderer

logger = logging.getLogger("simulator")


class AttackSimulationEnv(gym.Env):
    """
    Handles reinforcement learning matters.
    """

    NO_ACTION = "no action"

    sim: AttackSimulator
    attacker: Agent

    def __init__(self, config: Union[EnvConfig, dict]):

        super(AttackSimulationEnv, self).__init__()

        if isinstance(config, dict):
            config = EnvConfig(**config)

        self.rng, self.env_seed = get_rng(config.seed)

        # process configuration, leave the graph last, as it may destroy env_config
        self.config = config
        self.attacker_class = ATTACKERS[config.attacker]
        self.save_graphs = config.save_graphs
        self.save_logs = config.save_logs

        self.sim = AttackSimulator(self.config, self.rng)
        self.rewards = np.array(self.sim.g.reward_params)

        # An observation informs the defender of
        # a) which services are turned on; and,
        # b) which attack steps have been successfully taken
        self.dim_observations = self.sim.num_assets + self.sim.num_attack_steps
        # Using a Box instead of Tuple((Discrete(2),) * self.dim_observations)
        # avoids potential preprocessor issues with Ray
        # (cf. https://github.com/ray-project/ray/issues/8600)
        self.observation_space = spaces.Box(0, 1, shape=(self.dim_observations,), dtype="int8")

        # The defender action space allows to disable any one service or leave all unchanged
        self.num_actions = self.sim.num_assets + 1
        self.action_space = spaces.Discrete(self.num_actions)

        self.episode_count = 0

        self.done = False
        self.reward = None
        self.renderer = None

        self.episode_id = self._get_episode_id()

        self.max_reward = 0
        self.attack_start_time = 0

    def _create_attacker(self):
        return self.attacker_class(
            dict(
                attack_graph=self.sim.g,
                ttc=self.sim.ttc_remaining,
                rewards=self.rewards,
                random_seed=self.config.seed + self.episode_count,
            )
        )

    def _get_episode_id(self):
        # TODO connect this with ray run id/wandb run id instead of random seed.
        return f"{self.config.seed}_{self.episode_count}"

    def reset(self):
        logger.debug("Starting new simulation. (%d)", self.episode_id)
        self.done = False

        self.episode_count += 1
        self.episode_id = self._get_episode_id()
        self.attack_start_time = int(self.rng.exponential(self.config.attack_start_time))
        self.max_reward = sum(self.rewards)

        # Set up a new simulation environment
        self.sim = AttackSimulator(self.config, self.rng)
        self.rewards = np.array(self.sim.g.reward_params)

        # Set up a new attacker
        self.attacker = self._create_attacker()

        return self.sim.observe()

    def reward_function(self, attacker_reward, mode="simple"):
        service_state = self.sim.service_state

        reward = 0

        if mode == "simple":
            reward = sum(service_state) - attacker_reward
        elif mode == "capped":
            reward = self.max_reward
            reward -= attacker_reward
            reward -= sum(1 - service_state)
            reward = max(0, reward / self.max_reward)
        elif mode == "delayed":
            if self.done:
                reward = sum(service_state) - sum(self.rewards[self.sim.attack_state])
            else:
                reward = sum(service_state)
        else:
            raise Exception("Invalid Reward Method.")

        return reward

    def step(self, action):
        assert 0 <= action < self.num_actions

        self.done = False
        attacker_reward = 0

        # reserve 0 for no action
        if action:
            # decrement to obtain index
            service = action - 1
            self.done = self.sim.defense_action(service)

        if not self.done:
            # Check if the attack has started
            if self.sim.time >= self.attack_start_time:
                # Obtain attacker action, this _can_ be 0 for no action
                attack_index = self.attacker.act(self.sim.attack_surface) - 1
                self.done = self.attacker.done
                assert -1 <= attack_index < self.sim.num_attack_steps

                if attack_index != -1:
                    self.done = self.sim.attack_action(attack_index)
                    attacker_reward = self.rewards[attack_index]

                # TODO: placeholder, none of the current attackers learn...
                # self.attacker.update(attack_surface, attacker_reward, self.done)

        self.sim.step()

        # compute defender reward
        # positive reward for maintaining services online (1 unit per service)
        # negative reward for the attacker's gains (as measured by `attacker_reward`)
        # FIXME: the reward for maintaining services is _very_ low

        self.reward = self.reward_function(attacker_reward, mode=self.config.reward_mode)

        compromised_steps = self.sim.compromised_steps
        compromised_flags = self.sim.compromised_flags
        current_step, ttc_remaining = self.sim.current_attack_step()

        info = {
            "time": self.sim.time,
            "attack_surface": self.sim.attack_surface,
            "current_step": current_step,
            "ttc_remaining_on_current_step": ttc_remaining,
            "compromised_steps": compromised_steps,
            "compromised_flags": compromised_flags,
        }

        if self.done:
            logger.debug("Attacker done")
            logger.debug("Compromised steps: %s", compromised_steps)
            logger.debug("Compromised flags: %s", compromised_flags)

        return self.sim.observe(), self.reward, self.done, info

    def render(self, mode="human", subdir=None):
        if not self.renderer:
            self.renderer = AttackSimulationRenderer(
                self.sim,
                self.episode_count,
                self.rewards,
                subdir=subdir,
                save_graph=self.config.save_graphs,
                save_logs=self.config.save_logs,
            )
        self.renderer.render(self.reward)
        return True

    def interpret_action_probabilities(self, action_probabilities):
        keys = [self.NO_ACTION] + self.sim.g.service_names
        return {key: value for key, value in zip(keys, action_probabilities)}

    def seed(self, seed: int = None) -> List[int]:
        if seed is None:
            return [self.env_seed]

        self.rng, self.env_seed = get_rng(seed)

        return [seed]
