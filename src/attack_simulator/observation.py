from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import UINT


@dataclass
class Observation:
    ids_observation: NDArray[np.int8]
    attack_surface: NDArray[np.int8]
    defense_state: NDArray[np.int8]
    ttc_remaining: NDArray[UINT]
    attack_state: NDArray[np.int8]


@dataclass
class Info:
    time: UINT
    num_compromised_steps: UINT
    num_compromised_flags: UINT
    perc_compromised_flags: float
    perc_compromised_steps: float
    perc_defenses_activated: float
    num_observed_alerts: UINT


def obs_to_dict(obs: Observation) -> dict:
    return {
        "ids_observation": np.array(obs.ids_observation),
        "attack_surface": np.array(obs.attack_surface),
        "defense_state": np.array(obs.defense_state),
        "ttc_remaining": np.array(obs.ttc_remaining),
        "attack_state": np.array(obs.attack_state),
    }
