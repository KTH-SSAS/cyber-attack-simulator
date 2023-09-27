import dataclasses
import json
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from maturin import import_hook

from .roles import Defender, Attacker
from ..constants import AGENT_ATTACKER, AGENT_DEFENDER
from ..agents import ATTACKERS
from ..mal.observation import Info, Observation
from ..mal.sim import Simulator
from ..renderer.renderer import AttackSimulationRenderer
from ..utils.config import EnvConfig, GraphConfig, SimulatorConfig
from ..utils.rng import get_rng


# install the import hook with default settings
import_hook.install(bindings="pyo3")

from ..rusty_sim import RustAttackSimulator  # noqa: E402

logger = logging.getLogger("simulator")


def uptime_reward(
    _defender_action, defense_state: NDArray[np.int8], defense_costs: NDArray[np.int8]
) -> float:
    # Defender is rewarded each timestep for each defense that has been not used
    return sum(defense_costs * defense_state)


def downtime_penalty(
    _defender_action, defense_state: NDArray[np.int8], defense_costs: NDArray[np.int8]
) -> float:
    # Defender is penalized each timestep for each defense that has been used
    return sum(defense_costs * defense_state)


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

    defender_obs = Defender.get_obs(sim_obs)
    attacker_obs = Attacker.get_obs(sim_obs)

    return {AGENT_DEFENDER: defender_obs, AGENT_ATTACKER: attacker_obs}


class EnvironmentState:
    def __init__(self):
        self.cumulative_rewards = {AGENT_DEFENDER: 0.0, AGENT_ATTACKER: 0.0}
        self.reward = {AGENT_ATTACKER: 0.0, AGENT_DEFENDER: 0.0}
        self.terminated = {AGENT_DEFENDER: False, AGENT_ATTACKER: False, "__all__": False}
        self.truncated = {AGENT_DEFENDER: False, AGENT_ATTACKER: False, "__all__": False}


class AttackSimulationEnv(MultiAgentEnv):
    """Handles reinforcement learning matters."""

    NO_ACTION = "no action"

    # attacker: Agent
    sim: Simulator
    last_obs: Observation

    def __init__(self, config: EnvConfig, render_mode: str | None = None):
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

        # Set the seed for the simulator.
        sim_config = dataclasses.replace(sim_config, seed=config.seed)

        self.reward_function = reward_funcs[config.reward_mode]
        self.sim = RustAttackSimulator(
            json.dumps(sim_config.to_dict()), graph_config.filename, graph_config.vocab_filename
        )  # noqa: F821
        self.rng, self.env_seed = get_rng(config.seed)
        self.config = config
        self.render_mode = render_mode

        x: Tuple[Observation, Info] = self.sim.reset()
        obs: Observation = Observation.from_rust(x[0])
        info: Info = x[1]

        actions = self.sim.actions
        num_actions = len(actions)
        # terminate_action_idx = actions["terminate"]
        wait_action_idx = actions["wait"]

        num_defenses = self.sim.num_defense_steps
        num_attacks = self.sim.num_attack_steps
        num_nodes = len(obs.nodes)
        num_edges = len(obs.edges)

        self.d_impact = np.array(self.sim.defender_impact)
        self.a_impact = np.array(self.sim.attacker_impact)

        self.observation_space: spaces.Dict = self.define_observation_space(
            num_defenses, num_nodes, num_edges, num_actions
        )
        self.action_space: spaces.Dict = self.define_action_space(
            num_defenses, num_attacks, num_actions
        )

        self.state = EnvironmentState()
        self._agent_ids = [AGENT_DEFENDER, AGENT_ATTACKER]
        self._action_space_in_preferred_format = True
        self._observation_space_in_preferred_format = True
        self._obs_space_in_preferred_format = True
        self.episode_count = (
            -1
        )  # Start episode count at -1 since it will be incremented the first time reset is called.
        self.render_env = config.save_graphs or config.save_logs
        self.renderer: Optional[AttackSimulationRenderer] = None
        self.reset_render = True
        self.n_nodes = num_nodes
        self.n_actions = num_actions
        # self.terminate_action_idx = terminate_action_idx
        self.wait_action_idx = wait_action_idx
        self.screen = None
        self.vocab = self.sim.vocab
        self.reverse_vocab = [None] * len(self.vocab)
        for key, value in self.vocab.items():
            self.reverse_vocab[value] = key
        super().__init__()

    @staticmethod
    def define_action_space(num_defenses, num_attacks, num_actions) -> spaces.Discrete:
        return spaces.Dict(
            {
                AGENT_DEFENDER: spaces.MultiDiscrete([num_actions, num_defenses + num_attacks]),
                AGENT_ATTACKER: spaces.MultiDiscrete([num_actions, num_defenses + num_attacks]),
            }
        )

    @staticmethod
    def define_observation_space(
        num_defenses, n_nodes: int, n_edges, num_actions: int
    ) -> spaces.Dict:
        return spaces.Dict(
            {
                AGENT_DEFENDER: Defender.obs_space(num_actions, n_nodes, n_edges),
                AGENT_ATTACKER: Attacker.obs_space(num_actions, n_nodes),
            }
        )

    def get_observation_shapes(self):
        return {
            agent: {obs_key: space.shape}
            for agent, a_space in self.observation_space.spaces.items()
            for obs_key, space in a_space.spaces.items()
        }

    @staticmethod
    def create_renderer(episode_count: int, config: EnvConfig) -> AttackSimulationRenderer:
        return AttackSimulationRenderer(
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
        sim_obs = Observation.from_rust(sim_obs)

        self.episode_count = episode_count
        self.reset_render = True
        self.state = EnvironmentState()
        self.rng = rng
        self.env_seed = env_seed

        # Reset the simulator

        self.last_obs = sim_obs

        agent_obs = get_agent_obs(sim_obs)
        agent_info = self.get_agent_info(info)

        if self.render_env:
            self.render()

        return agent_obs, agent_info

    def observation_space_sample(self, agent_ids: list = None):

        agent_ids = self._agent_ids if agent_ids is None else agent_ids

        return {agent_id: self.observation_space[agent_id].sample() for agent_id in agent_ids}

    def action_space_sample(self, agent_ids: list = None):

        agent_ids = self._agent_ids if agent_ids is None else agent_ids

        return {agent_id: self.action_space[agent_id].sample() for agent_id in agent_ids}

    def action_space_contains(self, x) -> bool:
        return all(self.action_space[agent_id].contains(x[agent_id]) for agent_id in x)

    def observation_space_contains(self, x) -> bool:
        return all(self.observation_space[agent_id].contains(x[agent_id]) for agent_id in x)

    def get_agent_info(self, info: Info) -> Dict[str, Any]:
        infos = {
            AGENT_DEFENDER: Defender.get_info(info),
            AGENT_ATTACKER: Attacker.get_info(info),
        }

        for key, entry in infos.items():
            entry[f"{key}_cumulative_reward"] = self.state.cumulative_rewards[key]


        return infos

    def step(self, action_dict) -> Tuple[Dict, float, bool, dict]:

        truncated = {agent_id: False for agent_id in self._agent_ids}

        truncated["__all__"] = False

        # Convert numpy arrays to python tuples
        action_dict = {agent_id: tuple(action) for agent_id, action in action_dict.items()}
        # terminated = {key: value == self.terminate_action_idx for key, value in action_dict.items()}
        # terminated["__all__"] = all(terminated.values())

        sim_obs, info = self.sim.step(action_dict)
        sim_obs = Observation.from_rust(sim_obs)

        obs = get_agent_obs(sim_obs)
        infos = self.get_agent_info(info)
        rewards = {}
        state = sim_obs.nodes
        rewards[AGENT_ATTACKER] = np.sum(self.a_impact[state])
        rewards[AGENT_DEFENDER] = np.sum(self.d_impact[state])


        terminated = self.state.terminated
        terminated[AGENT_ATTACKER] = Attacker.done(sim_obs)
        terminated["__all__"] = Attacker.done(sim_obs)

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

    def render(self) -> bytes:

        # """Render a frame of the environment."""
        # if not self.render_env:
        #     return True

        # if self.reset_render:
        #     self.renderer = self.create_renderer(self.episode_count, self.config)
        #     self.reset_render = False

        # if isinstance(self.renderer, AttackSimulationRenderer):
        #     self.renderer.render(self.last_obs,
        #     self.state.reward[AGENT_DEFENDER], self.done)

        screen_width = 1000
        screen_height = 1000

        try:
            import pygame
            import graphviz
            import PIL.Image
            import io
        except ImportError as e:
            raise RuntimeError(
                "Missing render dependency"
            ) from e
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (screen_width, screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((screen_width, screen_height))


        self.screen.fill((255, 255, 255))
        pygame.display.set_caption("Attack Simulation")
        ## Render the graph
        graphviz_code = self.sim.render()
        graphviz_graph = graphviz.Source(graphviz_code)
        graphviz_graph.format = "png"
        #graphviz_graph.render("graphviz_graph")
        graphviz_graph = PIL.Image.open(io.BytesIO(graphviz_graph.pipe()))
        graphviz_graph = graphviz_graph.resize((screen_width, screen_height))
        graphviz_graph = pygame.image.fromstring(
            graphviz_graph.tobytes(), graphviz_graph.size, graphviz_graph.mode
        )
        
        self.screen.blit(graphviz_graph, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            ) 

    def interpret_action_probabilities(
        self, defense_names, action_probabilities: np.ndarray
    ) -> dict:
        keys = [self.NO_ACTION] + defense_names
        return {key: value for key, value in zip(keys, action_probabilities)}


def register_rllib_env() -> str:
    name = "AttackSimulationEnv"
    # Register the environment in the registry
    def env_creator(env_config: Dict) -> AttackSimulationEnv:
        config_data: EnvConfig = EnvConfig(**env_config)
        return AttackSimulationEnv(config_data)

    register_env(name, env_creator)
    return name
