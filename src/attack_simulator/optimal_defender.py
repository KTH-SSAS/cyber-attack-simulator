from typing import Dict, List, Optional, Tuple, Union

import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorStructType, TensorType
from torch import Tensor


class TripwireDefender(Algorithm):
    # _allow_unknown_configs = True

    def get_default_policy_class(self, config):
        return TripwirePolicy


class TripwirePolicy(Policy):
    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)
        self.num_outputs = action_space.n
        self.compromised_steps = torch.zeros(1, 1)
        self.attack_steps_by_defense_steps: List[List[int]] = config["defense_steps"]
        self.num_defense_steps = len(self.attack_steps_by_defense_steps)

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

        # obs = input_dict["obs"]
        # sim_state: Tensor = obs["sim_obs"].type(torch.FloatTensor)

        obs = restore_original_dimensions(
            torch.Tensor(input_dict["obs"]), self.observation_space, "numpy"
        )

        sim_state: Tensor = obs["sim_obs"].type(torch.FloatTensor)

        attack_state = sim_state[:, self.num_defense_steps :]
        defense_state = sim_state[:, : self.num_defense_steps]

        defenses_to_activate = torch.zeros(attack_state.shape[0], self.num_defense_steps)

        for b in range(attack_state.shape[0]):
            for d_s, defendable_steps in enumerate(self.attack_steps_by_defense_steps):
                if defense_state[b, d_s] == 1:  # if the defense is available
                    for a_s in defendable_steps:
                        if attack_state[b, a_s] == 1:
                            defenses_to_activate[b, d_s] = 1

        policy_out = torch.zeros(attack_state.shape[0], self.num_outputs)
        for b, defense in enumerate(defenses_to_activate):
            if sum(defense) == 0:
                policy_out[b, 0] = 1
            else:
                policy_out[b, 1:] = defense

        return torch.argmax(policy_out, dim=1), [], {}

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

    def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int = 0) -> int:
        return 0

    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
        pass
