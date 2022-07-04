import logging
from typing import Dict, List, Optional, Tuple, Type

import gym
import gym.spaces as spaces
import numpy as np
from ray.tune.registry import register_env

from attack_simulator.agents import ATTACKERS
from attack_simulator.agents.agent import Agent
from attack_simulator.config import EnvConfig
from attack_simulator.rng import get_rng
from attack_simulator.sim import AttackSimulator

from .renderer import AttackSimulationRenderer

logger = logging.getLogger("simulator")


class AttackSimulationEnv(gym.Env):
    """Handles reinforcement learning matters."""

    NO_ACTION = "no action"

    sim: AttackSimulator
    attacker: Agent

    def __init__(self, config: EnvConfig):

        super().__init__()

        self.rng, self.env_seed = get_rng(config.seed)

        # process configuration, leave the graph last, as it may destroy env_config
        self.config = config
        self.attacker_class: Type[Agent] = ATTACKERS[config.attacker]

        self.render_env = config.save_graphs or config.save_logs

        self.sim = AttackSimulator(self.config, self.rng)

        # Include reward for wait action (-1)
        self.attacker_rewards = np.concatenate((np.array(self.sim.g.reward_params), np.zeros(1)))

        # An observation informs the defender of
        # a) which services are turned on; and,
        # b) which attack steps have been successfully taken
        self.dim_observations = self.sim.num_defense_steps + self.sim.num_attack_steps
        # Using a Box instead of Tuple((Discrete(2),) * self.dim_observations)
        # avoids potential preprocessor issues with Ray
        # (cf. https://github.com/ray-project/ray/issues/8600)
        
        self.num_actions = self.sim.num_defense_steps + 1
        self.observation_space = spaces.Dict(
            {
                "action_mask": spaces.Box(0, 1, shape=(self.num_actions,), dtype=np.int8),
            "sim_state": spaces.Box(0, 1, shape=(self.dim_observations,), dtype=np.int8),
            }
        )

        # The defender action space allows to disable any one service or leave all unchanged
        self.action_space = spaces.Discrete(self.num_actions)

        # Start this at -1 since it will be incremented by reset.
        self.episode_count = -1

        self.done = False
        self.defender_reward = 0.0

        self.renderer: Optional[AttackSimulationRenderer] = None
        self.reset_render = True

        self.max_reward = 0.0
        self.attack_start_time = 0

    def _create_attacker(self) -> Agent:
        return self.attacker_class(
            dict(
                simulator=self.sim,
                attack_graph=self.sim.g,
                ttc=self.sim.ttc_remaining,
                rewards=self.attacker_rewards,
                random_seed=self.env_seed + self.episode_count,
            )
        )

    @staticmethod
    def create_renderer(
        sim: AttackSimulator, episode_count: int, config: EnvConfig
    ) -> AttackSimulationRenderer:
        return AttackSimulationRenderer(
            sim,
            config.run_id,
            episode_count,
            save_graph=config.save_graphs,
            save_logs=config.save_logs,
        )

    def reset(self) -> Dict:
        self.done = False

        self.reset_render = True

        self.episode_count += 1

        self.attack_start_time = int(self.rng.exponential(self.config.attack_start_time))
        self.max_reward = sum(self.attacker_rewards)

        # Set up a new simulation environment
        self.sim = AttackSimulator(self.config, self.rng)
        self.attacker_rewards = self.sim.g.reward_params

        # Set up a new attacker
        self.attacker = self._create_attacker()

        obs = {
            "sim_state": self.sim.observe(),
            "action_mask": self.get_action_mask(),
        }

        if self.render_env:
            self.render()

        return obs

    def reward_function(self, attacker_reward: float, mode: str = "simple") -> float:
        """Calculates the defender reward.

        Only 'simple' works at the moment.
        """

        active_defense_costs = self.sim.g.defense_costs * self.sim.defense_state
        upkeep_reward = sum(active_defense_costs)
        reward = 0.0

        if mode == "simple":
            reward = upkeep_reward - attacker_reward
        elif mode == "capped":
            reward = self.max_reward
            # Penalty for attacker gains
            reward -= attacker_reward
            # Penalty for defenses activated
            reward -= np.sum(self.sim.g.defense_costs * (not self.sim.defense_state))
            reward = max(0, reward / self.max_reward)
        elif mode == "delayed":
            if self.done:
                reward = upkeep_reward - sum(self.attacker_rewards[self.sim.attack_state])
            else:
                reward = upkeep_reward
        else:
            raise Exception("Invalid Reward Method.")

        return reward

    def step(self, action: int) -> Tuple[Dict, float, bool, dict]:
        assert 0 <= action < self.num_actions

        done = False
        attacker_reward = 0

        # offset action to take the wait action into account
        defender_action = action - 1
        self.sim.defense_action(defender_action)

        # Check if the attack has started
        if self.sim.time >= self.attack_start_time:
            # Obtain attacker action, this _can_ be 0 for no action
            attacker_action = self.attacker.act(self.sim.attack_surface) - 1 
            done |= self.attacker.done
            assert -1 <= attacker_action < self.sim.num_attack_steps

            done |= self.sim.attack_action(attacker_action)
            attacker_reward = self.attacker_rewards[attacker_action]

        done |= self.sim.step()

        # compute defender reward
        # positive reward for maintaining services online (1 unit per service)
        # negative reward for the attacker's gains (as measured by `attacker_reward`)
        # FIXME: the reward for maintaining services is _very_ low

        self.defender_reward = self.reward_function(attacker_reward, mode=self.config.reward_mode)

        compromised_steps = self.sim.compromised_steps
        compromised_flags = self.sim.compromised_flags
        current_step, ttc_remaining = self.sim.current_attack_step()

        info = {
            "time": self.sim.time,
            "attack_surface": self.sim.attack_surface,
            "current_step": current_step,
            "ttc_remaining_on_current_step": ttc_remaining,
            "attacker_reward": attacker_reward,
            "attacker_start_time": self.attack_start_time,
            "num_compromised_steps": len(compromised_steps),
            "num_compromised_flags": len(compromised_flags),
            "num_defenses_activated": sum(self.sim.defense_state),
            "num_services_online": sum(self.sim.service_state),
        }

        if done:
            logger.debug("Compromised steps: %s", compromised_steps)
            logger.debug("Compromised flags: %s", compromised_flags)

        self.done = done

        obs = {
            "sim_state": self.sim.observe(),
            "action_mask": self.get_action_mask(),
        }

        return obs, self.defender_reward, done, info

    def get_action_mask(self) -> np.ndarray:
        action_mask = np.ones(self.num_actions, dtype=np.int8)
        action_mask[1:] = self.sim.defense_state
        return action_mask

    def render(self, mode: str = "human") -> bool:
        """Render a frame of the environment."""

        if not self.render_env:
            return True
        
        if self.reset_render:
            self.renderer = self.create_renderer(self.sim, self.episode_count, self.config)
            self.reset_render = False

        if isinstance(self.renderer, AttackSimulationRenderer):
        self.renderer.render(self.defender_reward, self.done)

        return True

    def interpret_action_probabilities(self, action_probabilities: np.ndarray) -> dict:
        keys = [self.NO_ACTION] + self.sim.g.defense_names
        return {key: value for key, value in zip(keys, action_probabilities)}

    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is None:
            return [self.env_seed]

        self.rng, self.env_seed = get_rng(seed)

        return [seed]

def register_rllib_env() -> str:
    name = "AttackSimulationEnv"
    # Register the environment in the registry
    def env_creator(env_config: Dict) -> AttackSimulationEnv:

        config_data: EnvConfig = EnvConfig(**env_config)
        return AttackSimulationEnv(config_data)

    register_env(name, env_creator)
    return name