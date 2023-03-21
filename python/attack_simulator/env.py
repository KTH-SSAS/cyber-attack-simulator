import dataclasses
import logging
from typing import Any, Dict, Optional, Tuple, Type, Union

from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from rust_sim import RustAttackSimulator

from .agents import ATTACKERS
from .agents.agent import Agent
from .config import EnvConfig, GraphConfig, SimulatorConfig
from .constants import (
    ACTION_TERMINATE,
    AGENT_ATTACKER,
    AGENT_DEFENDER,
    UINT,
    special_actions,
)
from .graph import AttackGraph
from .observation import Info, Observation
from .rng import get_rng
from .sim import AttackSimulator

from .renderer import AttackSimulationRenderer
from .rust_wrapper import rust_sim_init

logger = logging.getLogger("simulator")


def select_random_attacker(rng: np.random.Generator):
    return list(ATTACKERS.values())[rng.integers(0, len(ATTACKERS))]


class AttackSimulationEnv(MultiAgentEnv):
    """Handles reinforcement learning matters."""

    NO_ACTION = "no action"

    # attacker: Agent
    sim: Union[AttackSimulator, RustAttackSimulator]
    attacker_rewards: NDArray[np.float32]

    def __init__(self, config: EnvConfig):
        self.rng, self.env_seed = get_rng(config.seed)
        self.config = config
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

        # Environment seed overrides the simulator seed
        sim_config = dataclasses.replace(sim_config)

        self.state_cache = None

        self.g: AttackGraph = AttackGraph(graph_config)

        if self.config.backend == "python":
            sim_init_func = AttackSimulator
        elif self.config.backend == "rust":
            sim_init_func = rust_sim_init
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

        self.sim = sim_init_func(sim_config, self.g)

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
                        "ids_observation": spaces.Box(
                            0, 1, shape=(self.dim_observations,), dtype=np.int8
                        ),
                    }
                ),
                AGENT_ATTACKER: spaces.Dict(
                    {
                        "attack_surface": spaces.Box(
                            0, 1, shape=(self.g.num_attacks,), dtype=np.int8
                        ),
                        "ttc_remaining": spaces.Box(
                            0, np.iinfo(UINT).max, shape=(self.g.num_attacks,), dtype=UINT
                        ),
                        "defense_state": spaces.Box(
                            0, 1, shape=(self.g.num_defenses,), dtype=np.int8
                        ),
                    }
                ),
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
        self.cumulative_rewards = {AGENT_DEFENDER: 0.0, AGENT_ATTACKER: 0.0}
        super().__init__()

        # Start episode count at -1 since it will be incremented the first time reset is called.
        self.episode_count = -1

        # process configuration, leave the graph last, as it may destroy env_config

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
        if seed is None:
            seed = self.config.seed

        super().reset(seed=seed)

        self.rng, self.env_seed = get_rng(seed)

        self.done = False

        self.reset_render = True

        self.terminateds = {AGENT_DEFENDER: False, AGENT_ATTACKER: False, "__all__": False}
        self.truncateds = {AGENT_DEFENDER: False, AGENT_ATTACKER: False, "__all__": False}

        self.episode_count += 1

        self.cumulative_rewards = {AGENT_DEFENDER: 0.0, AGENT_ATTACKER: 0.0}

        self.defender_reward = 0

        # Reset the simulator
        sim_obs, info = self.sim.reset(self.env_seed + self.episode_count)

        self.state_cache = sim_obs

        # Include reward for wait action (-1)
        self.attacker_rewards = np.concatenate((np.array(self.g.reward_params), np.zeros(1)))

        if self.config.attacker == "mixed":
            self.attacker_class = select_random_attacker(self.rng)

        agent_obs = self.get_agent_obs(sim_obs)
        agent_info = self.get_agent_info(info)

        if self.render_env:
            self.render()

        return agent_obs, agent_info

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

    def get_agent_obs(self, sim_obs: Observation) -> Dict[str, Any]:
        defender_obs = {
            "ids_observation": np.array(sim_obs.ids_observation, dtype=np.int8),
            "action_mask": self.get_action_mask(sim_obs.defense_state),
        }

        attacker_obs = {
            "attack_surface": np.array(sim_obs.attack_surface, dtype=np.int8),
            "defense_state": np.array(sim_obs.defense_state, dtype=np.int8),
            "ttc_remaining": np.array(sim_obs.ttc_remaining, dtype=np.uint64),
        }

        return {AGENT_DEFENDER: defender_obs, AGENT_ATTACKER: attacker_obs}

    def get_agent_info(self, info: Info) -> Dict[str, Any]:
        infos = {
            AGENT_DEFENDER: {
                "perc_defenses_activated": info.perc_defenses_activated,
                "num_observed_alerts": info.num_observed_alerts,
            },
            AGENT_ATTACKER: {
                "num_compromised_steps": info.num_compromised_steps,
                "percent_compromised_steps": info.perc_compromised_steps,
            },
        }

        for key, entry in infos.items():
            entry["cumulative_reward"] = self.cumulative_rewards[key]

        return infos

    def step(self, action_dict) -> Tuple[Dict, float, bool, dict]:
        attacker_reward = 0

        defender_action = action_dict[AGENT_DEFENDER]
        attacker_action = action_dict[AGENT_ATTACKER]

        old_attack_state = self.state_cache.attack_state

        sim_obs, info = self.sim.step(action_dict)
        self.state_cache = sim_obs

        attack_surface = sim_obs.attack_surface

        new_compromised_steps = (
            np.array(sim_obs.attack_state, dtype=np.int8)
            - np.array(old_attack_state, dtype=np.int8)
        ).clip(0, 1)
        attacker_done = not any(attack_surface) or attacker_action == ACTION_TERMINATE
        attacker_reward = np.sum(self.attacker_rewards[new_compromised_steps])

        self.sum_attacker_reward += attacker_reward

        defense_state = sim_obs.defense_state

        defender_penalty = self.reward_function(
            defender_action,
            defense_costs=self.defense_costs,
            defense_state=defense_state,
            mode=self.config.reward_mode,
        )
        self.sum_defender_penalty += defender_penalty

        self.defender_reward = defender_penalty - attacker_reward

        obs = self.get_agent_obs(sim_obs)

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

        infos = self.get_agent_info(info)

        if terminated["__all__"] or truncated["__all__"]:
            self.done = True

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
