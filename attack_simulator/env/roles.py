from ..mal.observation import Info, Observation
import numpy as np
from gymnasium import spaces
from typing import Any, Dict
from gymnasium.spaces import Box

BIG_INT = 2**63 - 2


class Defender:
    @staticmethod
    def get_info(info: Info, obs: Observation) -> Dict[str, Any]:
        action_mask = (
            obs.defender_possible_actions,
            obs.defender_possible_objects,
        )
        return {
            "action_mask": action_mask,
            "perc_defenses_activated": info.perc_defenses_activated,
        }

    @staticmethod
    def get_obs(obs: Observation) -> Dict[str, Any]:
        return {
            "observation": obs.defender_observation,
            "asset": obs.assets,
            "asset_id": obs.asset_ids,
            "ttc_remaining": obs.ttc_remaining,
            "step_name": obs.names,
            "edges": obs.edges,
        }


class Attacker:
    @staticmethod
    def get_info(info: Info, obs: Observation) -> Dict[str, Any]:
        action_mask = (
            obs.attacker_possible_actions,
            obs.attacker_possible_objects,
        )
        return {
            "action_mask": action_mask,
            "num_compromised_steps": info.num_compromised_steps,
            "perc_compromised_steps": info.perc_compromised_steps,
            "perc_compromised_flags": info.perc_compromised_flags,
            "sum_ttc_remaining": info.sum_ttc,
        }

    @staticmethod
    def get_obs(obs: Observation) -> Dict[str, Any]:
        return {
            "observation": obs.attacker_observation,
            "asset": obs.assets,
            "asset_id": obs.asset_ids,
            "step_name": obs.names,
            "ttc_remaining": obs.ttc_remaining,
            "edges": obs.edges,
        }

    @staticmethod
    def done(obs: Observation) -> bool:
        attack_surface = obs.attacker_possible_objects
        return not any(attack_surface)
