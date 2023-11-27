import dataclasses
import json
import logging
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from maturin import import_hook

from .. import examplemanager
from .roles import Defender, Attacker
from ..constants import AGENT_ATTACKER, AGENT_DEFENDER
from ..mal.observation import Info, Observation
from ..utils.config import EnvConfig, SimulatorConfig
from ..utils.rng import get_rng


# install the import hook with default settings
import_hook.install(bindings="pyo3")

from ..rusty_sim import RustAttackSimulator  # noqa: E402

logger = logging.getLogger("simulator")


def defender_reward(obs: Observation) -> int:
    return np.sum(obs.defender_reward)


def attacker_reward(obs: Observation) -> int:
    return np.sum(obs.attacker_reward)


def get_agent_obs(agents: list, sim_obs: Observation) -> Dict[str, Any]:
    obs_funcs = {AGENT_DEFENDER: Defender.get_obs, AGENT_ATTACKER: Attacker.get_obs}

    return {key: obs_funcs[key](sim_obs) for key in agents}


class EnvironmentState:
    def __init__(self, agent_ids: list) -> None:
        self.cumulative_rewards = {k: 0.0 for k in agent_ids}
        self.reward: Dict[str, int] = {k: 0 for k in agent_ids}
        self.terminated: Dict[str, bool] = {k: False for k in agent_ids}
        self.terminated["__all__"] = False
        self.truncated: Dict[str, bool] = {k: False for k in agent_ids}
        self.truncated["__all__"] = False


class AttackSimulationEnv:
    """Handles reinforcement learning matters."""

    NO_ACTION = "no action"

    # attacker: Agent
    last_obs: Dict[str, Any]

    def __init__(self, config: EnvConfig, render_mode: str | None = None):
        sim_config = (
            config.sim_config
            if isinstance(config.sim_config, SimulatorConfig)
            else SimulatorConfig(**config.sim_config)
        )

        graph_filename = examplemanager.get_paths_to_graphs()[config.graph_name]

        self.sim = RustAttackSimulator(
            json.dumps(sim_config.to_dict()), graph_filename, config.vocab_filename
        )  # noqa: F821
        self.config = config
        self.render_mode = render_mode

        x: Tuple[Observation, Info] = self.sim.reset()
        obs: Observation = Observation.from_rust(x[0])

        actions = self.sim.actions
        num_actions = len(actions)

        num_nodes = len(obs.state)
        num_edges = len(obs.edges)

        self.observation_space: spaces.Dict = self.define_observation_space(
            num_nodes, num_edges, num_actions
        )
        self.action_space: spaces.Dict = self.define_action_space(num_nodes, num_actions)

        self._agent_ids = (
            [AGENT_DEFENDER, AGENT_ATTACKER] if not config.attacker_only else [AGENT_ATTACKER]
        )
        self.state = EnvironmentState(self._agent_ids)
        self._action_space_in_preferred_format = True
        self._observation_space_in_preferred_format = True
        self._obs_space_in_preferred_format = True
        # Start episode count at -1 since it will be incremented the first time reset is called.
        self.episode_count = -1
        self.screen: Optional[Any] = None
        self.vocab = self.sim.vocab
        self.reverse_vocab = [None] * len(self.vocab)
        for key, value in self.vocab.items():
            self.reverse_vocab[value] = key
        super().__init__()

    @staticmethod
    def define_action_space(n_nodes: int, num_actions: int) -> spaces.Dict:
        return spaces.Dict(
            {
                AGENT_DEFENDER: spaces.MultiDiscrete([num_actions, n_nodes]),
                AGENT_ATTACKER: spaces.MultiDiscrete([num_actions, n_nodes]),
            }
        )

    @staticmethod
    def define_observation_space(n_nodes: int, n_edges: int, num_actions: int) -> spaces.Dict:
        return spaces.Dict(
            {
                AGENT_DEFENDER: Defender.obs_space(num_actions, n_nodes, n_edges),
                AGENT_ATTACKER: Attacker.obs_space(num_actions, n_nodes),
            }
        )

    def get_observation_shapes(self) -> Dict[str, Any]:
        return {
            agent: {obs_key: space.shape}
            for agent, a_space in self.observation_space.spaces.items()
            for obs_key, space in a_space.spaces.items()
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sim_obs, info = self.sim.reset(seed)
        sim_obs = Observation.from_rust(sim_obs)
        self.state = EnvironmentState(self._agent_ids)
        agent_obs = get_agent_obs(self._agent_ids, sim_obs)
        agent_info = self.get_agent_info(self._agent_ids, info)
        self.last_obs = agent_obs
        self.episode_count += 1
        return agent_obs, agent_info

    def observation_space_sample(self, agent_ids: list = None) -> Dict[str, Any]:
        agent_ids = self._agent_ids if agent_ids is None else agent_ids

        return {agent_id: self.observation_space[agent_id].sample() for agent_id in agent_ids}

    def action_space_sample(self, agent_ids: list = None) -> Dict[str, Any]:
        agent_ids = self._agent_ids if agent_ids is None else agent_ids

        return {agent_id: self.action_space[agent_id].sample() for agent_id in agent_ids}

    def action_space_contains(self, x: list) -> bool:
        return all(self.action_space[agent_id].contains(x[agent_id]) for agent_id in x)

    def observation_space_contains(self, x: list) -> bool:
        return all(self.observation_space[agent_id].contains(x[agent_id]) for agent_id in x)

    def get_agent_info(self, agent_ids: list, info: Info) -> Dict[str, Dict[str, Any]]:
        info_funcs = {
            AGENT_DEFENDER: Defender.get_info,
            AGENT_ATTACKER: Attacker.get_info,
        }

        infos = {key: info_funcs[key](info) for key in agent_ids}

        # for key, entry in infos.items():
        #     entry["return"] = self.state.cumulative_rewards[key]

        return infos

    def step(
        self, action_dict: dict
    ) -> tuple[
        Dict[str, Any],
        Dict[str, SupportsFloat],
        Dict[str, bool],
        Dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        truncated = {agent_id: False for agent_id in self._agent_ids}

        truncated["__all__"] = False

        # Convert numpy arrays to python tuples
        action_dict = {agent_id: tuple(action) for agent_id, action in action_dict.items()}
        action_dict = {
            agent_id: (a, None) if a == 0 else (a, n) for agent_id, (a, n) in action_dict.items()
        }
        # terminated = {key: value == self.terminate_action_idx for key, value in action_dict.items()}
        # terminated["__all__"] = all(terminated.values())

        sim_obs, info = self.sim.step(action_dict)
        sim_obs = Observation.from_rust(sim_obs)

        obs = get_agent_obs(self._agent_ids, sim_obs)
        infos = self.get_agent_info(self._agent_ids, info)
        rewards = {}

        done_funcs = {
            AGENT_DEFENDER: Attacker.done,  # Defender is done when attacker is done
            AGENT_ATTACKER: Attacker.done,
        }

        reward_funcs = {
            AGENT_DEFENDER: defender_reward,
            AGENT_ATTACKER: attacker_reward,
        }

        terminated = {key: done_funcs[key](sim_obs) for key in self._agent_ids}

        rewards = {
            key: reward_funcs[key](sim_obs) if not terminated[key] else 0 for key in self._agent_ids
        }

        terminated["__all__"] = Attacker.done(sim_obs)
        self.state.reward = rewards
        for key, value in rewards.items():
            self.state.cumulative_rewards[key] += value
        self.state.terminated = terminated
        self.state.truncated = truncated
        self.last_obs = obs

        return obs, rewards, terminated, truncated, infos

    @property
    def done(self) -> bool:
        return self.state.terminated["__all__"] or self.state.truncated["__all__"]

    def close(self) -> None:
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

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
            raise RuntimeError("Missing render dependency") from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((screen_width, screen_height))

        self.screen.fill((255, 255, 255))
        pygame.display.set_caption("Attack Simulation")
        ## Render the graph
        graphviz_code = self.sim.render()
        graphviz_graph = graphviz.Source(graphviz_code)
        graphviz_graph.engine = "dot"
        graphviz_graph.format = "png"
        graphviz_graph.renderer = "cairo"
        graphviz_graph.formatter = "cairo"
        # graphviz_graph.render("graphviz_graph")
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
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        return

    def interpret_action_probabilities(
        self, defense_names: list, action_probabilities: np.ndarray
    ) -> dict:
        keys = [self.NO_ACTION] + defense_names
        return {key: value for key, value in zip(keys, action_probabilities)}
