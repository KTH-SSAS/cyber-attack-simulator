import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
from torch import Tensor, nn


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
        hidden_dim = 256
        num_hidden_layers = 2
        activation_func = nn.Tanh

        hidden_layers = []
        prev_layer_size = sim_space.shape[0]
        for _ in range(num_hidden_layers):
            hidden_layers.append(nn.Linear(prev_layer_size, hidden_dim))
            hidden_layers.append(activation_func())
            prev_layer_size = hidden_dim

        self.embedding_func = nn.Sequential(
            *hidden_layers,
        )

        self.policy_fn = nn.Linear(hidden_dim, num_outputs)
        self.value_fn = nn.Linear(hidden_dim, 1)
        
    def forward(self, input_dict, state, seq_lens):

        obs = input_dict["obs"]

        sim_state: Tensor = obs["sim_state"].type(torch.FloatTensor)
        action_mask: Tensor = obs["action_mask"].type(torch.FloatTensor)

        if torch.cuda.is_available():
            sim_state = sim_state.cuda()
            action_mask = action_mask.cuda()


        embedding = self.embedding_func(sim_state)

        policy_out = self.policy_fn(embedding)
        value_out = self.value_fn(embedding)
        self._value_out = value_out

        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return policy_out + inf_mask, state

    def value_function(self):
        return self._value_out.flatten()

    def import_from_h5(self, h5_file: str) -> None:
        return super().import_from_h5(h5_file)
