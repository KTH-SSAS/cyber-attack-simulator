from re import A
import numpy as np
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
        sim_space = obs_space.original_space.spaces["sim_state"]
        self.action_space = action_space
        self.num_outputs = num_outputs

        self.num_defense_steps = kwargs["num_defense_steps"]

        self.policy_fn = nn.Linear(sim_space.shape[0], self.action_space.n)
        self.value_fn = nn.Linear(sim_space.shape[0], 1)

    def forward(self, input_dict, state, seq_lens):

        obs = input_dict["obs"]

        sim_state: Tensor = obs["sim_state"].type(torch.FloatTensor)
        action_mask: Tensor = obs["action_mask"].type(torch.FloatTensor)

        if torch.cuda.is_available():
            sim_state = sim_state.cuda()
            action_mask = action_mask.cuda()

        policy_out = self.policy_fn(sim_state)
        value_out = self.value_fn(sim_state)
        self._value_out = value_out

        inf_mask = torch.max(torch.log(action_mask), torch.ones_like(action_mask) * -1e10)
        
        return policy_out + inf_mask, state

    def value_function(self):
        return self._value_out.flatten()

    def import_from_h5(self, h5_file: str) -> None:
        return super().import_from_h5(h5_file)
