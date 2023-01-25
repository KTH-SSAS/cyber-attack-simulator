import logging
from typing import Dict, Optional, Tuple, Type

import gymnasium.spaces as spaces
import numpy as np
from numpy.typing import NDArray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from attack_simulator.agents import ATTACKERS
from attack_simulator.agents.agent import Agent
from attack_simulator.config import EnvConfig, GraphConfig, SimulatorConfig
from attack_simulator.constants import (
    ACTION_TERMINATE,
    AGENT_ATTACKER,
    AGENT_DEFENDER,
    special_actions,
)
from attack_simulator.graph import AttackGraph
from attack_simulator.rng import get_rng
from attack_simulator.sim import AttackSimulator

from .renderer import AttackSimulationRenderer

logger = logging.getLogger("simulator")


def select_random_attacker(rng: np.random.Generator):
    return list(ATTACKERS.values())[rng.integers(0, len(ATTACKERS))]


class AttackSimulationEnv(MultiAgentEnv):
    """Handles reinforcement learning matters."""

    NO_ACTION = "no action"

    # attacker: Agent
    sim: AttackSimulator
    attacker_rewards: NDArray[np.float32]

    def __init__(self, config: EnvConfig):
        self.rng, self.env_seed = get_rng(config.seed)

        graph_config = (
            config.graph_config
            if isinstance(config.graph_config, GraphConfig)
            else GraphConfig(**config.graph_config)
        )

        sim_config = (
            config.sim_config
            if isinstance(config.sim_config, SimulatorConfig)
            else SimulatorConfig(**config.sim_config)
        )

        self.state_cache = {}

        self.g: AttackGraph = AttackGraph(graph_config)
        self.sim = AttackSimulator(sim_config, self.g, self.env_seed)
        # self.sim = Simulator(json.dumps(sim_config.replace(seed=self.env_seed).to_dict()), graph_config.filename)

        # An observation informs the defender of
        # a) which services are turned on; and,
        # b) which attack steps have been successfully taken
        self.dim_observations = self.g.num_defenses + self.g.num_attacks
        self.defense_costs = self.g.defense_costs

        self.num_actions = self.g.num_defenses + len(special_actions)
        self.observation_space = spaces.Dict(
            {
                AGENT_DEFENDER: spaces.Dict(
                    {
                        "action_mask": spaces.Box(0, 1, shape=(self.num_actions,), dtype=np.int8),
                        "sim_obs": spaces.Box(0, 1, shape=(self.dim_observations,), dtype=np.int8),
                    }
                ),
                AGENT_ATTACKER: spaces.Box(0, 1, shape=(self.g.num_attacks,), dtype=np.int8),
            }
        )

        # The defender action space allows to disable any one service or leave all unchanged
        self.action_space = spaces.Dict(
            {
                AGENT_DEFENDER: spaces.Discrete(self.num_actions),
                AGENT_ATTACKER: spaces.Discrete(self.g.num_attacks + len(special_actions)),
            }
        )

        self._agent_ids = [AGENT_DEFENDER, AGENT_ATTACKER]
        self._action_space_in_preferred_format = True
        self._observation_space_in_preferred_format = True
        self._obs_space_in_preferred_format = True
        super().__init__()

        # Start episode count at -1 since it will be incremented the first time reset is called.
        self.episode_count = -1

        # process configuration, leave the graph last, as it may destroy env_config
        self.config = config
        self.attacker_class: Type[Agent] = (
            ATTACKERS[config.attacker]
            if config.attacker != "mixed"
            else select_random_attacker(self.rng)
        )

        self.render_env = config.save_graphs or config.save_logs

        self.terminateds = {AGENT_DEFENDER: False, AGENT_ATTACKER: False}
        self.truncateds = {AGENT_DEFENDER: False, AGENT_ATTACKER: False}

        self.renderer: Optional[AttackSimulationRenderer] = None
        self.reset_render = True
        self.defender_reward = 0.0
        self.sum_attacker_reward = 0
        self.sum_defender_penalty = 0
        self.done = False

    def _create_attacker(self, simulator: AttackSimulator, attack_graph: AttackGraph) -> Agent:
        return self.attacker_class(
            dict(
                simulator=simulator,
                attack_graph=attack_graph,
                ttc=simulator.ttc_remaining,
                rewards=self.attacker_rewards,
                random_seed=self.env_seed + self.episode_count,
            )
        )

    @staticmethod
    def create_renderer(
        graph: AttackGraph, episode_count: int, config: EnvConfig
    ) -> AttackSimulationRenderer:
        return AttackSimulationRenderer(
            graph,
            config.run_id,
            episode_count,
            save_graph=config.save_graphs,
            save_logs=config.save_logs,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.rng, self.env_seed = get_rng(seed)

        self.done = False

        self.reset_render = True

        self.terminateds = {AGENT_DEFENDER: False, AGENT_ATTACKER: False, "__all__": False}
        self.truncateds = {AGENT_DEFENDER: False, AGENT_ATTACKER: False, "__all__": False}

        self.episode_count += 1

        self.sum_attacker_reward = 0
        self.sum_defender_penalty = 0
        self.defender_reward = 0

        # Reset the simulator
        obs, info = self.sim.reset(self.env_seed + self.episode_count)

        # Include reward for wait action (-1)
        self.attacker_rewards = np.concatenate((np.array(self.g.reward_params), np.zeros(1)))

        if self.config.attacker == "mixed":
            self.attacker_class = select_random_attacker(self.rng)

        # Set up a new attacker
        # self.attacker = self._create_attacker(self.sim, self.g)

        # obs = {
        #     "action_mask": self.get_action_mask(obs.defense_state),
        #     "sim_obs": np.array(obs.defense_state + obs.attack_state, dtype=np.int8),
        # }

        obs = {
            "sim_obs": obs["defender_obs"],
            "action_mask": self.get_action_mask(info["defense_state"]),
        }

        if self.render_env:
            self.render()

        return {AGENT_DEFENDER: obs, AGENT_ATTACKER: info["attack_surface"]}, info

    def observation_space_sample(self, agent_ids: list = None):
        return {
            AGENT_DEFENDER: self.observation_space[AGENT_DEFENDER].sample(),
            AGENT_ATTACKER: self.observation_space[AGENT_ATTACKER].sample(),
        }

    def action_space_sample(self, agent_ids: list = None):
        return {
            AGENT_DEFENDER: self.action_space[AGENT_DEFENDER].sample(),
            AGENT_ATTACKER: self.action_space[AGENT_ATTACKER].sample(),
        }

    def action_space_contains(self, x) -> bool:
        return all(self.action_space[agent_id].contains(x[agent_id]) for agent_id in x)

    def observation_space_contains(self, x) -> bool:
        return all(self.observation_space[agent_id].contains(x[agent_id]) for agent_id in x)

    def reward_function(
        self,
        defender_action: int,
        defense_costs: NDArray[np.int64],
        defense_state: NDArray[np.int8],
        mode: str = "simple",
    ) -> float:
        """Calculates the defender reward.

        Only 'simple' works at the moment.
        """

        # The costs of used defense steps

        reward = 0.0
        if mode == "uptime-reward":
            # Defender is rewarded each timestep for each defense that has been not used
            defense_reward = sum(defense_costs * defense_state)
            reward = defense_reward
        elif mode == "downtime-penalty":
            # Defender is penalized each timestep for each defense that has been used
            defense_penalty = sum(defense_costs * (np.logical_not(defense_state)))
            reward = -defense_penalty
        elif mode == "defense-penalty":
            # Defender is penalized once when a defense is used
            reward = -(defense_costs[defender_action]) if defender_action != -1 else 0
        else:
            raise Exception(f"Invalid reward method: {mode}.")

        return reward

    def step(self, action_dict) -> Tuple[Dict, float, bool, dict]:
        attacker_reward = 0

        # offset action to take the wait action into account
        defender_action = action_dict[AGENT_DEFENDER]
        defender_obs = self.sim.action(defender_action, AGENT_DEFENDER)

        # Obtain attacker action, this _can_ be 0 for no action
        # attacker_action = self.attacker.act(self.sim.attacker_observation)
        # done |= self.attacker.done
        # assert -1 <= attacker_action < self.sim.num_attack_steps

        attacker_action = action_dict[AGENT_ATTACKER]
        attacker_obs = self.sim.action(attacker_action, AGENT_ATTACKER)

        self.state_cache = attacker_obs

        attack_surface = attacker_obs["attack_surface"]
        compromised_steps = attacker_obs["affected_steps"]

        attacker_done = not any(attack_surface) or attacker_action == ACTION_TERMINATE
        attacker_reward = np.sum(self.attacker_rewards[compromised_steps])

        self.sum_attacker_reward += attacker_reward

        _, info = self.sim.step()

        defense_state = info["defense_state"]

        defender_penalty = self.reward_function(
            defender_action,
            defense_costs=self.defense_costs,
            defense_state=defense_state,
            mode=self.config.reward_mode,
        )
        self.sum_defender_penalty += defender_penalty

        self.defender_reward = defender_penalty - attacker_reward

        obs = {
            AGENT_DEFENDER: {
                "sim_obs": defender_obs["defender_obs"],
                "action_mask": self.get_action_mask(defense_state),
            },
            AGENT_ATTACKER: info["attack_surface"],
        }

        rewards = {AGENT_DEFENDER: self.defender_reward, AGENT_ATTACKER: attacker_reward}
        terminated = {
            AGENT_DEFENDER: attacker_done,
            AGENT_ATTACKER: attacker_done,
            "__all__": attacker_done,
        }
        truncated = {
            AGENT_DEFENDER: False,
            AGENT_ATTACKER: False,
            "__all__": False,
        }

        if terminated["__all__"] or truncated["__all__"]:
            self.done = True
            info = info | self.sim.summary()
            info["sum_attacker_reward"] = self.sum_attacker_reward
            info["sum_defender_penalty"] = self.sum_defender_penalty

        infos = {AGENT_DEFENDER: info, AGENT_ATTACKER: info}

        return obs, rewards, terminated, truncated, infos

    def get_action_mask(self, defense_state: NDArray[np.int8]) -> NDArray[np.int8]:
        action_mask = np.ones(self.num_actions, dtype=np.int8)
        action_mask[len(special_actions) :] = defense_state
        return action_mask

    def render(self) -> bool:
        """Render a frame of the environment."""

        if not self.render_env:
            return True

        if self.reset_render:
            self.renderer = self.create_renderer(self.g, self.episode_count, self.config)
            self.reset_render = False

        if isinstance(self.renderer, AttackSimulationRenderer):
            self.renderer.render(self.state_cache, self.defender_reward, self.done)

        return True

    def interpret_action_probabilities(self, action_probabilities: np.ndarray) -> dict:
        keys = [self.NO_ACTION] + self.g.defense_names
        return {key: value for key, value in zip(keys, action_probabilities)}


def register_rllib_env() -> str:
    name = "AttackSimulationEnv"
    # Register the environment in the registry
    def env_creator(env_config: Dict) -> AttackSimulationEnv:
        config_data: EnvConfig = EnvConfig(**env_config)
        return AttackSimulationEnv(config_data)

    register_env(name, env_creator)
    return name
