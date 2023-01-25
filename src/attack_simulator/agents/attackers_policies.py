from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tree
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorStructType, TensorType

from attack_simulator.agents.searchers import BreadthFirstAttacker, DepthFirstAttacker


class NonLearningAttackerPolicy(Policy):
    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)
        self.num_outputs = action_space.n
        self.observation_space = obs_space

    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List["Episode"]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        return [], [], {}

    def learn_on_batch(self, samples):
        return {}

    def compute_gradients(self, postprocessed_batch: SampleBatch):
        pass

    def get_weights(self):
        return {}

    def set_weights(self, weights) -> None:
        pass

    def init_view_requirements(self):
        super().init_view_requirements()
        # Disable for_training and action attributes for SampleBatch.INFOS column
        # since it can not be properly batched.
        vr = self.view_requirements[SampleBatch.INFOS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False

    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(
                    lambda s: s[None], self.observation_space.sample()
                ),
            }
        )


class BreadthFirstPolicy(NonLearningAttackerPolicy):
    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)
        self.attacker = BreadthFirstAttacker({"seed": config["seed"]})

    def compute_actions_from_input_dict(
        self,
        input_dict: Union[SampleBatch, Dict[str, TensorStructType]],
        explore: bool = None,
        timestep: Optional[int] = None,
        episodes: Optional[List["Episode"]] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        attack_surface = input_dict["obs"]
        action = self.attacker.act(attack_surface)
        return [action], [], {}


class DepthFirstPolicy(NonLearningAttackerPolicy):
    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)
        self.attacker = DepthFirstAttacker({"seed": config["seed"]})

    def compute_actions_from_input_dict(
        self,
        input_dict: Union[SampleBatch, Dict[str, TensorStructType]],
        explore: bool = None,
        timestep: Optional[int] = None,
        episodes: Optional[List["Episode"]] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        attack_surface = input_dict["obs"]
        action = self.attacker.act(attack_surface)
        return [action], [], {}


class DedicatedRandomPolicy(NonLearningAttackerPolicy):
    """Selects a random attack step from the attack surface and works on it
    until it is compromised."""

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)
        self.current_step = None
        self.rng = np.random.default_rng(config["seed"])

    def compute_actions_from_input_dict(
        self,
        input_dict: Union[SampleBatch, Dict[str, TensorStructType]],
        explore: bool = None,
        timestep: Optional[int] = None,
        episodes: Optional[List["Episode"]] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        attack_surface = input_dict["obs"]
        valid_attack_indices = np.flatnonzero(attack_surface)
        if self.current_step is None or self.current_step not in valid_attack_indices:
            self.current_step = (
                self.rng.choice(valid_attack_indices) if len(valid_attack_indices) > 0 else 0
            )

        return [self.current_step], [], {}
