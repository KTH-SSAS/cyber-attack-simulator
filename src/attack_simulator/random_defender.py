import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
from torch import Tensor
import numpy as np
from torch import nn

def register_rllib_model():
    name = "RandomDefenderModel"
    ModelCatalog.register_custom_model(name, RandomDefenderModel)
    return name


class RandomDefenderModel(TorchModelV2, nn.Module):
    """Policy for the agent agent."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.num_outputs = num_outputs
        self.value_fn = nn.Linear(obs_space.original_space.spaces["sim_state"].shape[0], 1)
        self.value_out = torch.ones(1, 1)

    def forward(self, input_dict, state, seq_lens):

        obs = input_dict["obs"]
        
        action_mask: Tensor = obs["action_mask"].type(torch.FloatTensor)

        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        policy_out = torch.rand(self.num_outputs) + inf_mask

        self.value_out = self.value_fn(obs["sim_state"].type(torch.FloatTensor))

        return policy_out, state

    def value_function(self):
        return self.value_out.flatten()

    def import_from_h5(self, h5_file: str) -> None:
        return super().import_from_h5(h5_file)
