from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..constants import UINT


@dataclass
class Observation:
    #attacker role
    attacker_observation: NDArray[np.int8]
    attacker_possible_actions: NDArray[np.int8]
    attacker_possible_objects: NDArray[np.int8]
    # defender role
    defender_observation: NDArray[np.int8]
    defender_possible_actions: NDArray[np.int8]
    defender_possible_objects: NDArray[np.int8]
    # common state
    state: NDArray[np.int8]
    ttc_remaining: NDArray[UINT]
    # Info
    assets: NDArray[UINT]
    asset_ids: NDArray[UINT]
    names: NDArray[UINT]
    edges: NDArray[UINT]
    flags: NDArray[UINT]
    # Rewards
    attacker_reward: int
    defender_reward: int

    @classmethod
    def from_rust(cls, obs) -> "Observation":
        assets, asset_ids, names = zip(*obs.step_info)
        return Observation(
            attacker_observation=np.array(obs.attacker_observation, dtype=np.int8),
            defender_observation=np.array(obs.defender_observation, dtype=np.int8),
            defender_possible_objects=np.array(obs.defender_possible_objects, dtype=np.int8),
            attacker_possible_objects=np.array(obs.attacker_possible_objects, dtype=np.int8),
            defender_possible_actions=np.array(obs.defender_possible_actions, dtype=np.int8),
            attacker_possible_actions=np.array(obs.attacker_possible_actions, dtype=np.int8),
            state=np.array(obs.state, dtype=np.int8),
            assets=np.array(assets, dtype=np.int64),
            asset_ids=np.array(asset_ids, dtype=np.int64),
            names=np.array(names, dtype=np.int64),
            ttc_remaining=np.array(obs.ttc_remaining, dtype=np.int64),
            edges=np.array(obs.edges, dtype=np.int64),
            flags=np.array(obs.flags, dtype=np.int64),
            attacker_reward=obs.attacker_reward,
            defender_reward=obs.defender_reward,
        )


@dataclass
class Info:
    time: int
    num_compromised_steps: int
    num_compromised_flags: int
    perc_compromised_flags: float
    perc_compromised_steps: np.double
    perc_defenses_activated: np.double
    sum_ttc: UINT


def obs_to_dict(obs: Observation) -> dict:
    return {
        "ids_observation": np.array(obs.defender_observation),
        "node_surface": np.array(obs.attacker_possible_objects),
        "defense_state": np.array(obs.defense_state),
        "ttc_remaining": np.array(obs.ttc_remaining),
        "attack_state": np.array(obs.attack_state),
    }
