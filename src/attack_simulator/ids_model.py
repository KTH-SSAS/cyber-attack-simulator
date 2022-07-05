import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import Tensor, nn
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX


def register_rllib_model():
    name = "DefenderModel"
    ModelCatalog.register_custom_model(name, DefenderModel)
    return name


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


        self.policy_fn = nn.Linear(sim_space.shape[0], num_outputs)
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

        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return policy_out + inf_mask, state

    def value_function(self):
        return self._value_out.flatten()

    def import_from_h5(self, h5_file: str) -> None:
        return super().import_from_h5(h5_file)
