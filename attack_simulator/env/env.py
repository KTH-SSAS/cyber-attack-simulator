import dataclasses
import json
import logging
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box
from gymnasium.core import RenderFrame
from numpy.typing import NDArray

from .. import examplemanager
from .roles import Defender, Attacker
from ..constants import AGENT_ATTACKER, AGENT_DEFENDER
from ..mal.observation import Info, Observation
from ..utils.config import EnvConfig, SimulatorConfig
from ..utils.rng import get_rng

from ..rusty_sim import RustAttackSimulator  # noqa: E402

logger = logging.getLogger("simulator")


def obs_to_graphviz(obs: dict, info: Optional[dict] = None) -> str:
    translations = info.get("translated", {}) if info is not None else {}
    graphviz_code = "digraph G {\n"
    for i, (state, asset, asset_id, step_name) in enumerate(
        zip(obs["observation"], obs["asset"], obs["asset_id"], obs["step_name"])
    ):
        color = "red" if state == 1 else "white"
        node_label = (
            f"{asset}:{asset_id}:{step_name}" if translations == {} else translations["nodes"][i]
        )
        graphviz_code += f'\t"{i}" [label="{node_label}", style="filled", fillcolor="{color}"]\n'  # has-type: ignore
    for parent, child in obs["edges"]:
        graphviz_code += f'\t"{parent}" -> "{child}"\n'
    graphviz_code += "}"
    return graphviz_code


def defender_reward(obs: Observation) -> float:
    return float(obs.defender_reward)


def attacker_reward(obs: Observation) -> float:
    return float(obs.attacker_reward)


def get_agent_obs(agents: list, sim_obs: Observation) -> Dict[str, Any]:
    obs_funcs = {AGENT_DEFENDER: Defender.get_obs, AGENT_ATTACKER: Attacker.get_obs}

    return {key: obs_funcs[key](sim_obs) for key in agents}


BIG_INT = 2**63 - 2


def obs_space(n_actions: int, n_objects: int, n_edges: int, vocab_size: int) -> spaces.Dict:
    # n_features = 1  # TODO maybe merge some of the dict fields into a single array
    return spaces.Dict(
        {
            # "action_mask": spaces.Tuple(
            #     (
            #         Box(
            #             0,
            #             1,
            #             shape=(n_actions,),
            #             dtype=np.int8,
            #         ),
            #         Box(
            #             0,
            #             1,
            #             shape=(n_objects,),
            #             dtype=np.int8,
            #         ),
            #     )
            # ),
            "observation": Box(-1, 1, shape=(n_objects,), dtype=np.int8),
            "asset": Box(0, vocab_size, shape=(n_objects,), dtype=np.int64),
            "ttc_remaining": Box(
                0,
                BIG_INT,
                shape=(n_objects,),
                dtype=np.int64,
            ),
            "asset_id": Box(
                0, BIG_INT, shape=(n_objects,), dtype=np.int64
            ),  # TODO this should the max number of assets
            "step_name": Box(0, vocab_size, shape=(n_objects,), dtype=np.int64),
            "edges": Box(
                0,
                n_objects,
                shape=(n_edges, 2),
                dtype=np.int64,
            ),
        }
    )


class EnvironmentState:
    def __init__(self, agent_ids: list) -> None:
        self.cumulative_rewards = {k: 0.0 for k in agent_ids}
        self.reward: Dict[str, float] = {k: 0.0 for k in agent_ids}
        self.terminated: Dict[str, bool] = {k: False for k in agent_ids}
        # self.terminated["__all__"] = False
        self.truncated: Dict[str, bool] = {k: False for k in agent_ids}
        # self.truncated["__all__"] = False


class AttackSimulationEnv:
    """Handles reinforcement learning matters."""

    NO_ACTION = "no action"
    clock = None
    # Start episode count at -1 since it will be incremented the first time reset is called.
    episode_count = -1
    _action_space_in_preferred_format = True
    _observation_space_in_preferred_format = True
    _obs_space_in_preferred_format = True
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    last_obs: Dict[str, Any]
    screen: Optional[Any] = None

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
        vocab = self.sim.vocab
        num_actions = len(actions)

        num_nodes = len(obs.state)
        num_edges = len(obs.edges)

        defenses = set(np.flatnonzero(obs.defender_possible_objects))
        if config.undirected_defenses:
            new_edges = self.get_reverse_edges(obs.edges, defenses)
            num_edges = num_edges + len(new_edges)

        self.observation_space: spaces.Dict = self.define_observation_space(
            num_nodes, num_edges, num_actions, len(vocab)
        )
        self.action_space: spaces.Dict = self.define_action_space(num_nodes, num_actions)

        self.possible_agents = [AGENT_ATTACKER, AGENT_DEFENDER]
        self._agent_ids = (
            [AGENT_DEFENDER, AGENT_ATTACKER] if not config.attacker_only else [AGENT_ATTACKER]
        )
        self.state = EnvironmentState(self._agent_ids)
        self.vocab = vocab
        self.reverse_vocab = [""] * len(self.vocab)
        self.defense_steps = defenses
        for key, value in self.vocab.items():
            self.reverse_vocab[value] = key
        super().__init__()

    @staticmethod
    def get_reverse_edges(edges: np.ndarray, defense_steps: set) -> np.ndarray:
        # Add reverse edges from the defense steps children to the defense steps
        # themselves
        new_edges = [np.array([c, p]) for p, c in edges if p in defense_steps]
        return np.stack(new_edges)

    @staticmethod
    def define_action_space(n_nodes: int, num_actions: int) -> spaces.Dict:
        return spaces.Dict(
            {
                AGENT_DEFENDER: spaces.MultiDiscrete([num_actions, n_nodes]),
                AGENT_ATTACKER: spaces.MultiDiscrete([num_actions, n_nodes]),
            }
        )

    @staticmethod
    def define_observation_space(
        n_nodes: int, n_edges: int, num_actions: int, vocab_size: int
    ) -> spaces.Dict:
        return spaces.Dict(
            {
                AGENT_DEFENDER: obs_space(num_actions, n_nodes, n_edges, vocab_size),
                AGENT_ATTACKER: obs_space(num_actions, n_nodes, n_edges, vocab_size),
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
        self._agent_ids = (
            [AGENT_DEFENDER, AGENT_ATTACKER] if not self.config.attacker_only else [AGENT_ATTACKER]
        )
        sim_obs, info = self.sim.reset(seed)
        sim_obs = Observation.from_rust(sim_obs)
        self.state = EnvironmentState(self._agent_ids)
        obs = get_agent_obs(self._agent_ids, sim_obs)
        if self.config.undirected_defenses and AGENT_DEFENDER in obs:
            edges = obs[AGENT_DEFENDER]["edges"]
            new_edges = self.get_reverse_edges(edges, self.defense_steps)
            edges = np.concatenate((edges, new_edges))
            obs[AGENT_DEFENDER]["edges"] = edges
        agent_info = self.get_agent_info(self._agent_ids, info, sim_obs)
        self.last_obs = obs
        self.episode_count += 1

        translated = {key: self.translate_obs(value) for key, value in obs.items()}

        for k, v in translated.items():
            agent_info[k]["translated"] = v

        return obs, agent_info

    def observation_space_sample(self, agent_ids: tuple = ()) -> Dict[str, Any]:
        agent_ids = self._agent_ids if agent_ids is None else agent_ids

        return {agent_id: self.observation_space[agent_id].sample() for agent_id in agent_ids}

    def action_space_sample(self, agent_ids: tuple = ()) -> Dict[str, Any]:
        agent_ids = self._agent_ids if agent_ids is None else agent_ids

        return {agent_id: self.action_space[agent_id].sample() for agent_id in agent_ids}

    def action_space_contains(self, x: tuple) -> bool:
        return all(self.action_space[agent_id].contains(x[agent_id]) for agent_id in x)

    def observation_space_contains(self, x: tuple) -> bool:
        return all(self.observation_space[agent_id].contains(x[agent_id]) for agent_id in x)

    def get_agent_info(
        self, agent_ids: list, info: Info, obs: Observation
    ) -> Dict[str, Dict[str, Any]]:
        info_funcs = {
            AGENT_DEFENDER: Defender.get_info,
            AGENT_ATTACKER: Attacker.get_info,
        }

        infos = {key: info_funcs[key](info, obs) for key in agent_ids}

        # for key, entry in infos.items():
        #     entry["return"] = self.state.cumulative_rewards[key]

        return infos

    def step(
        self, action_dict: dict
    ) -> tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        # truncated["__all__"] = False

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
        infos = self.get_agent_info(self._agent_ids, info, sim_obs)

        if self.config.undirected_defenses and AGENT_DEFENDER in obs:
            edges = obs[AGENT_DEFENDER]["edges"]
            new_edges = self.get_reverse_edges(edges, self.defense_steps)
            edges = np.concatenate((edges, new_edges))
            obs[AGENT_DEFENDER]["edges"] = edges

        done_funcs = {
            AGENT_DEFENDER: Attacker.done,  # Defender is done when attacker is done
            AGENT_ATTACKER: Attacker.done,
        }

        reward_funcs = {
            AGENT_DEFENDER: defender_reward,
            AGENT_ATTACKER: attacker_reward,
        }

        truncated = {agent_id: False for agent_id in self._agent_ids}
        terminated = {key: done_funcs[key](sim_obs) for key in self._agent_ids}
        rewards = {key: reward_funcs[key](sim_obs) for key in self._agent_ids}

        # terminated["__all__"] = Attacker.done(sim_obs)
        self.state.reward = rewards
        for key, value in rewards.items():
            self.state.cumulative_rewards[key] += value
        self.state.terminated = terminated
        self.state.truncated = truncated
        self.last_obs = obs

        translated = {key: self.translate_obs(value) for key, value in obs.items()}

        for k, v in translated.items():
            infos[k]["translated"] = v

        self._agent_ids = [k for k in self._agent_ids if not terminated[k] and not truncated[k]]

        return obs, rewards, terminated, truncated, infos

    def translate_obs(self, obs: dict) -> dict:
        nodes = [
            f"{self.reverse_vocab[int(a)]}:{int(i)}:{self.reverse_vocab[int(n)]}"
            for a, i, n in zip(obs["asset"], obs["asset_id"], obs["step_name"])
        ]
        return {
            "nodes": nodes,
            "edges": [(nodes[i], nodes[j]) for i, j in obs["edges"]],
        }

    # @property
    # def done(self) -> bool:
    #    return self.state.terminated["__all__"] or self.state.truncated["__all__"]

    def close(self) -> None:
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def render(self) -> RenderFrame | list[RenderFrame] | None:
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

        if self.clock is None:
            self.clock = pygame.time.Clock()

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
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        return

    def interpret_action_probabilities(
        self, defense_names: list, action_probabilities: np.ndarray
    ) -> dict:
        keys = [self.NO_ACTION] + defense_names
        return {key: value for key, value in zip(keys, action_probabilities)}
