import numpy as np
from numpy.typing import NDArray

# TODO: Move functionality from simulator to this class.
class IDS:
	def __init__(self) -> None:
		self.rng = np.random.default_rng()
		self.fnr = 0.0
		self.fpr = 0.0
		pass


class StrictIDS(IDS):
	def __init__(self) -> None:
		pass

	def observe(self, attack_state: NDArray[np.int8], defense_state: NDArray[np.int8]) -> NDArray[np.int8]:
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

		#false_negatives = self.attack_state & (probabilities >= self.false_negative)
		#false_positives = (1 - self.attack_state) & (probabilities <= self.false_positive)
		#detected = false_negatives | false_positives
		self.last_observation = observation
		return np.append(defense_state, observation)