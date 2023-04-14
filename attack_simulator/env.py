import dataclasses
import logging
from typing import Any, Dict, Optional, Tuple
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from .agents import ATTACKERS
from .config import EnvConfig, GraphConfig, SimulatorConfig
from .constants import (
    AGENT_ATTACKER,
    AGENT_DEFENDER,
    UINT,
)
from .graph import AttackGraph
from .observation import Info, Observation
from .rng import get_rng
from .sim import AttackSimulator, Simulator

from .renderer import AttackSimulationRenderer
from .rust_wrapper import rust_sim_init

logger = logging.getLogger("simulator")

sim_init_funcs = {
    "python": AttackSimulator,
    "rust": rust_sim_init,
}


def uptime_reward(
    _defender_action, defense_state: NDArray[np.int8], defense_costs: NDArray[np.int8]
) -> float:
    # Defender is rewarded each timestep for each defense that has been not used
    return sum(defense_costs * defense_state)


def downtime_penalty(
    _defender_action, defense_state: NDArray[np.int8], defense_costs: NDArray[np.int8]
) -> float:
    # Defender is penalized each timestep for each defense that has been used
    return -sum(defense_costs * (np.logical_not(defense_state)))


def defense_penalty(
    defender_action: int, _defense_state: NDArray[np.int8], defense_costs: NDArray[np.int8]
) -> float:
    # Defender is penalized once when a defense is used
    return -(defense_costs[defender_action]) if defender_action != -1 else 0


reward_funcs = {
    "uptime-reward": uptime_reward,
    "downtime-penalty": downtime_penalty,
    "defense-penalty": defense_penalty,
}


def select_random_attacker(rng: np.random.Generator):
    return list(ATTACKERS.values())[rng.integers(0, len(ATTACKERS))]

def get_agent_obs(sim_obs: Observation) -> Dict[str, Any]:
    defender_obs = {
        "ids_observation": np.array(sim_obs.ids_observation, dtype=np.int8),
        "action_mask": np.array(sim_obs.defender_action_mask),
    }

    attacker_obs = {
        "action_mask":   np.array(sim_obs.attacker_action_mask, dtype=np.int8),
        "defense_state": np.array(sim_obs.defense_state, dtype=np.int8),
        "ttc_remaining": np.array(sim_obs.ttc_remaining, dtype=np.uint64),
    }

    return {AGENT_DEFENDER: defender_obs, AGENT_ATTACKER: attacker_obs}
    
class EnvironmentState:
    def __init__(self):
        self.cumulative_rewards = {AGENT_DEFENDER: 0.0, AGENT_ATTACKER: 0.0}
        self.reward = {AGENT_ATTACKER: 0.0, AGENT_DEFENDER: 0.0}
        self.terminated = {AGENT_DEFENDER: False, AGENT_ATTACKER: False}
        self.truncated = {AGENT_DEFENDER: False, AGENT_ATTACKER: False}


class AttackSimulationEnv(MultiAgentEnv):
    """Handles reinforcement learning matters."""

    NO_ACTION = "no action"

    # attacker: Agent
    sim: Simulator
    last_obs: Observation

    def __init__(self, config: EnvConfig):

        graph_config = (
            config.graph_config
            if isinstance(config.graph_config, GraphConfig)
            else GraphConfig(**config.graph_config)
        )
        attack_graph = AttackGraph(graph_config)

        sim_config = (
            config.sim_config
            if isinstance(config.sim_config, SimulatorConfig)
            else SimulatorConfig(**config.sim_config)
        )

        # Include reward for wait action (-1)
        self.attacker_action_rewards = np.concatenate(
            (np.array(attack_graph.reward_params), np.zeros(1))
        )

        # Set the seed for the simulator.
        sim_config = dataclasses.replace(sim_config, seed=config.seed)

        self.reward_function = reward_funcs[config.reward_mode]
        self.sim = sim_init_funcs[config.backend](sim_config, attack_graph)
        self.rng, self.env_seed = get_rng(config.seed)
        self.defense_costs = attack_graph.defense_costs
        self.config = config

        self.num_special_actions = self.sim.num_special_actions
        
        self.terminate_action_idx = self.sim.terminate_action
        self.wait_action_idx = self.sim.wait_action

        self.observation_space = self.define_observation_space(
            attack_graph, self.num_special_actions
        )
        self.action_space = self.define_action_space(attack_graph, self.num_special_actions)

        self.state = EnvironmentState()
        self._agent_ids = [AGENT_DEFENDER, AGENT_ATTACKER]
        self._action_space_in_preferred_format = True
        self._observation_space_in_preferred_format = True
        self._obs_space_in_preferred_format = True

        self.g: AttackGraph = attack_graph
        self.episode_count = (
            -1
        )  # Start episode count at -1 since it will be incremented the first time reset is called.
        self.render_env = config.save_graphs or config.save_logs
        self.renderer: Optional[AttackSimulationRenderer] = None
        self.reset_render = True
        super().__init__()

    @staticmethod
    def define_action_space(graph: AttackGraph, num_special_actions) -> spaces.Discrete:
        return spaces.Dict(
            {
                AGENT_DEFENDER: spaces.Discrete(graph.num_defenses + num_special_actions),
                AGENT_ATTACKER: spaces.Discrete(graph.num_attacks + num_special_actions),
            }
        )

    @staticmethod
    def define_observation_space(graph: AttackGraph, num_special_actions: int) -> spaces.Dict:
        dim_observations = graph.num_defenses + graph.num_attacks
        return spaces.Dict(
            {
                AGENT_DEFENDER: spaces.Dict(
                    {
                        "action_mask": spaces.Box(
                            0, 1, shape=(graph.num_defenses + num_special_actions,), dtype=np.int8
                        ),
                        "ids_observation": spaces.Box(
                            0, 1, shape=(dim_observations,), dtype=np.int8
                        ),
                    }
                ),
                AGENT_ATTACKER: spaces.Dict(
                    {
                        "action_mask": spaces.Box(
                            0, 1, shape=(graph.num_attacks + num_special_actions,), dtype=np.int8
                        ),
                        "ttc_remaining": spaces.Box(
                            0, np.iinfo(UINT).max, shape=(graph.num_attacks,), dtype=UINT
                        ),
                        "defense_state": spaces.Box(
                            0, 1, shape=(graph.num_defenses,), dtype=np.int8
                        ),
                    }
                ),
            }
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
        if seed is None:
            seed = self.config.seed

        episode_count = self.episode_count + 1
        super().reset(seed=seed)

        rng, env_seed = get_rng(seed)
        sim_obs, info = self.sim.reset(env_seed + episode_count)

        self.episode_count = episode_count
        self.reset_render = True
        self.state = EnvironmentState()
        self.rng = rng
        self.env_seed = env_seed

        # Reset the simulator

        self.last_obs = sim_obs

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

    def get_agent_info(self, info: Info) -> Dict[str, Any]:
        infos = {
            AGENT_DEFENDER: {
                "perc_defenses_activated": info.perc_defenses_activated,
                "num_observed_alerts": info.num_observed_alerts,
                "defense_costs": self.g.defense_costs,
                "flag_costs": self.g.flag_rewards,
            },
            AGENT_ATTACKER: {
                "num_compromised_steps": info.num_compromised_steps,
                "perc_compromised_steps": info.perc_compromised_steps,
                "perc_compromised_flags": info.perc_compromised_flags,
            },
        }

        for key, entry in infos.items():
            entry[f"{key}_cumulative_reward"] = self.state.cumulative_rewards[key]

        return infos

    def step(self, action_dict) -> Tuple[Dict, float, bool, dict]:

        truncated = {
            AGENT_DEFENDER: False,
            AGENT_ATTACKER: False,
            "__all__": False,
        }

        defender_action = action_dict[AGENT_DEFENDER]
        attacker_action = action_dict[AGENT_ATTACKER]

        old_attack_state = self.last_obs.attack_state

        terminated = {key: value == self.terminate_action_idx for key, value in action_dict.items()}
        terminated["__all__"] = all(terminated.values())

        sim_obs, info = self.sim.step(action_dict)

        attack_surface = sim_obs.attack_surface

        new_compromised_steps = (
            np.array(sim_obs.attack_state, dtype=np.int8)
            - np.array(old_attack_state, dtype=np.int8)
        ).clip(0, 1)

        attacker_done = not any(attack_surface) or attacker_action == self.terminate_action_idx

        attacker_reward = np.sum(self.attacker_action_rewards[new_compromised_steps])

        defense_state = sim_obs.defense_state

        defender_penalty = self.reward_function(
            defender_action,
            defense_costs=self.defense_costs,
            defense_state=defense_state,
        )

        defender_reward = defender_penalty - attacker_reward

        obs = get_agent_obs(sim_obs)
        infos = self.get_agent_info(info)

        rewards = {AGENT_DEFENDER: defender_reward, AGENT_ATTACKER: attacker_reward}

        terminated[AGENT_ATTACKER] = attacker_done
        terminated["__all__"] = attacker_done

        self.state.reward = rewards
        for key, value in rewards.items():
            self.state.cumulative_rewards[key] += value
        self.state.terminated = terminated
        self.state.truncated = truncated
        self.last_obs = sim_obs

        return obs, rewards, terminated, truncated, infos

    @property
    def done(self):
        return self.state.terminated["__all__"] or self.state.truncated["__all__"]

    def render(self) -> bool:
        """Render a frame of the environment."""

        if not self.render_env:
            return True

        if self.reset_render:
            self.renderer = self.create_renderer(self.g, self.episode_count, self.config)
            self.reset_render = False

        if isinstance(self.renderer, AttackSimulationRenderer):
            self.renderer.render(self.last_obs, self.state.reward[AGENT_DEFENDER], self.done)

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
