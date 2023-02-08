from itertools import starmap
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from attack_simulator.constants import UINT


# TODO: Move functionality from simulator to this class.
class IDS:
    def __init__(self, seed: int, fnr: float, fpr: float) -> None:
        self.seed = seed
        self.fnr = fnr
        self.fpr = fpr


class ProbabilityIDS(IDS):
    def ids_function(self, obs: UINT, p: float) -> UINT:
        if obs == 1:
            if p <= self.fnr:
                return UINT(not obs)
            return obs
        if p <= self.fpr:
            return UINT(not obs)
        return obs

    def __call__(
        self, attack_state: NDArray[np.int8], probs: NDArray[np.float64]
    ) -> Tuple[NDArray, NDArray, NDArray]:

        noisy_obs = list(starmap(self.ids_function, zip(attack_state, probs)))

        fp = probs <= self.fpr
        fn = probs <= self.fnr

        return np.array(noisy_obs, dtype=np.int8), fp, fn


class StrictIDS(IDS):
    def __init__(self, seed: UINT) -> None:
        self.rng = np.random.default_rng(seed=seed)

    def observe(self, attack_state: NDArray[np.int8]) -> NDArray[np.int8]:
        """Observation of attack steps is subject to the true/false positive
        rates of an assumed underlying intrusion detection system Depending on
        the true and false positive rates for each step, ongoing attacks may
        not be reported, or non-existing attacks may be spuriously reported."""

        observation = attack_state.copy()

        num_alerts = attack_state.sum()
        num_silent = len(attack_state) - num_alerts
        num_false_positives = int(np.round(num_silent * self.fpr))
        num_false_negatives = int(np.round(num_alerts * self.fnr))

        # Set false positives
        false_positive_indices = self.rng.choice(
            np.flatnonzero(attack_state == 0), num_false_positives, replace=False
        )

        observation[false_positive_indices] = 1

        # Set false negatives
        false_negative_indices = self.rng.choice(
            np.flatnonzero(attack_state == 1), num_false_negatives, replace=False
        )

        observation[false_negative_indices] = 0

        # false_negatives = self.attack_state & (probabilities >= self.false_negative)
        # false_positives = (1 - self.attack_state) & (probabilities <= self.false_positive)
        # detected = false_negatives | false_positives
        self.last_observation = observation
        return observation
