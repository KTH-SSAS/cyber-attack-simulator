from re import A
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import Tensor, nn


class DefenderModel(TorchModelV2, nn.Module):
    """Policy for the agent agent."""

    _value_out: Tensor
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs

        self.num_defense_steps = kwargs["num_defense_steps"]

        self.policy_fn = nn.Linear(self.obs_space.shape[0], self.action_space.n)
        self.value_fn = nn.Linear(self.obs_space.shape[0], 1)

        action_mask = torch.ones(num_outputs) * -1e6
        action_mask[0] = 0 # 0 is the do nothing action, and we don't want to mask it
        self.action_mask = action_mask

    def forward(self, input_dict, state, seq_lens):

        obs = input_dict["obs"].type(torch.float32)

        defense_state = obs[:, :self.num_defense_steps]

        policy_out = self.policy_fn(obs)
        value_out = self.value_fn(obs)
        self._value_out = value_out

        # Mask out defenses that have alredy been used
        action_mask = self.action_mask.repeat(defense_state.shape[0], 1)
        defense_mask = (torch.logical_not(defense_state)) * 1e-6
        action_mask[:, 1:] = defense_mask
        
        return policy_out + action_mask, state

    def value_function(self):
        return self._value_out.flatten()

    def import_from_h5(self, h5_file: str) -> None:
        return super().import_from_h5(h5_file)
