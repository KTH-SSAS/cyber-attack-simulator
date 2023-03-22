from typing import Dict, List, Optional, Tuple, Union

import tree
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorStructType, TensorType


class RandomDefender(Algorithm):
    # _allow_unknown_configs = True

    def get_default_policy_class(self, config):
        return RandomPolicy


class RandomPolicy(Policy):

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
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def compute_actions_from_input_dict(
        self,
        input_dict: Union[SampleBatch, Dict[str, TensorStructType]],
        explore: bool = None,
        timestep: Optional[int] = None,
        episodes: Optional[List["Episode"]] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        return [self.action_space.sample() for _ in input_dict["obs"]], [], {}

    def learn_on_batch(self, samples):
        pass

    def compute_gradients(self, postprocessed_batch: SampleBatch):
        pass

    def get_weights(self):
        pass

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

    def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int = 0) -> int:
        return 0

    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
        pass
