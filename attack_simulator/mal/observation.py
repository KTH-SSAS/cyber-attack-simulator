from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..constants import UINT


@dataclass
class Observation:
    ids_observation: NDArray[np.int8]
    attack_surface: NDArray[np.int8]
    defense_surface: NDArray[np.int8]
    state: NDArray[np.int8]
    assets: NDArray[UINT]
    asset_ids: NDArray[UINT]
    names: NDArray[UINT]
    ttc_remaining: NDArray[UINT]
    defender_action_mask: NDArray[np.int8]
    attacker_action_mask: NDArray[np.int8]
    edges: NDArray[UINT]
    flags: NDArray[UINT]

    @classmethod
    def from_rust(cls, obs):
        state, assets, asset_ids, names = zip(*obs.nodes)
        return Observation(
            ids_observation=np.array(obs.ids_observation),
            attack_surface=np.array(obs.attack_surface),
            defense_surface=np.array(obs.defense_surface),
            state=np.array(state),
            assets=np.array(assets),
            asset_ids=np.array(asset_ids),
            names=np.array(names),
            ttc_remaining=np.array(obs.ttc_remaining),
            defender_action_mask=np.array(obs.defender_action_mask),
            attacker_action_mask=np.array(obs.attacker_action_mask),
            edges=np.array(obs.edges),
            flags=np.array(obs.flags),
        )


@dataclass
class Info:
    time: int
    num_compromised_steps: int
    num_compromised_flags: int
    perc_compromised_flags: float
    perc_compromised_steps: np.double
    perc_defenses_activated: np.double
    num_observed_alerts: UINT
    sum_ttc: UINT


def obs_to_dict(obs: Observation) -> dict:
    return {
        "ids_observation": np.array(obs.ids_observation),
        "node_surface": np.array(obs.attack_surface),
        "defense_state": np.array(obs.defense_state),
        "ttc_remaining": np.array(obs.ttc_remaining),
        "attack_state": np.array(obs.attack_state),
    }
