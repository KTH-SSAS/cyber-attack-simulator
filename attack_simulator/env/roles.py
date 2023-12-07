from ..mal.observation import Info, Observation
import numpy as np
from gymnasium import spaces
from typing import Any, Dict
from gymnasium.spaces import Box

BIG_INT = 2**63 - 2


class Defender:
    @staticmethod
    def obs_space(n_actions: int, n_objects: int, n_edges: int, vocab_size: int) -> spaces.Dict:
        n_features = 1  # TODO maybe merge some of the dict fields into a single array
        return spaces.Dict(
            {
                "action_mask": Box(0, 1, shape=(n_actions,), dtype=np.int8),
                "node_surface": Box(0, 1, shape=(n_objects,), dtype=np.int8),
                "observation": Box(0, 1, shape=(n_objects, n_features), dtype=np.int8),
                "asset": Box(0, vocab_size, shape=(n_objects, 1), dtype=np.int64),
                "asset_id": Box(
                    0, BIG_INT, shape=(n_objects, 1), dtype=np.int64
                ),  # TODO this should the max number of assets
                "step_name": Box(0, vocab_size, shape=(n_objects, 1), dtype=np.int64),
                "edges": Box(
                    0,
                    n_objects,
                    shape=(2, n_edges),
                    dtype=np.int64,
                ),
            }
        )

    @staticmethod
    def get_info(info: Info) -> Dict[str, Any]:
        return {
            "perc_defenses_activated": info.perc_defenses_activated,
        }

    @staticmethod
    def get_obs(obs: Observation) -> Dict[str, Any]:
        return {
            "action_mask": obs.defender_possible_actions,
            "node_surface": obs.defender_possible_objects,
            "observation": obs.defender_observation.reshape(-1, 1),
            "asset": obs.assets.reshape(-1, 1),
            "asset_id": obs.asset_ids.reshape(-1, 1),
            "step_name": obs.names.reshape(-1, 1),
            "edges": obs.edges.T,
        }


class Attacker:
    @staticmethod
    def get_info(info: Info) -> Dict[str, Any]:
        return {
            "num_compromised_steps": info.num_compromised_steps,
            "perc_compromised_steps": info.perc_compromised_steps,
            "perc_compromised_flags": info.perc_compromised_flags,
            "sum_ttc_remaining": info.sum_ttc,
        }

    @staticmethod
    def obs_space(n_actions: int, n_objects: int, n_edges: int, vocab_size: int) -> spaces.Dict:
        n_features = 1
        return spaces.Dict(
            {
                "action_mask": Box(
                    0,
                    1,
                    shape=(n_actions,),
                    dtype=np.int8,
                ),
                "node_surface": Box(
                    0,
                    1,
                    shape=(n_objects,),
                    dtype=np.int8,
                ),
                "ttc_remaining": Box(
                    0,
                    BIG_INT,
                    shape=(n_objects,),
                    dtype=np.int64,
                ),
                "observation": Box(
                    -1, 1, shape=(n_objects, n_features), dtype=np.int8
                ),  # -1 = unknown, 0 = not compromised, 1 = compromised
                "asset": Box(0, vocab_size, shape=(n_objects, 1), dtype=np.int64),
                "asset_id": Box(
                    0, BIG_INT, shape=(n_objects, 1), dtype=np.int64
                ),  # TODO this should the max number of assets
                "step_name": Box(0, vocab_size, shape=(n_objects, 1), dtype=np.int64),
                "nop_index": spaces.Discrete(n_actions),
                "edges": Box(
                    0,
                    n_objects,
                    shape=(2, n_edges),
                    dtype=np.int64,
                ),
            }
        )

    @staticmethod
    def get_obs(obs: Observation) -> Dict[str, Any]:
        return {
            "action_mask": obs.attacker_possible_actions,
            "node_surface": obs.attacker_possible_objects,
            "observation": obs.attacker_observation.reshape(-1, 1),
            "asset": obs.assets.reshape(-1, 1),
            "asset_id": obs.asset_ids.reshape(-1, 1),
            "step_name": obs.names.reshape(-1, 1),
            "ttc_remaining": obs.ttc_remaining,
            "edges": obs.edges.T,
            "nop_index": 0,
        }

    @staticmethod
    def done(obs: Observation) -> bool:
        attack_surface = obs.attacker_possible_objects
        return not any(attack_surface)
