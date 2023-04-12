from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium.spaces import Box
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
from torch import Tensor
from ..models.gnn import GNNRLAgent

class GNNDefenderModel(TorchModelV2, nn.Module):
    """Policy for the agent agent."""

    _value_out: Tensor

    def __init__(
        self, obs_space: Box, action_space, num_outputs, model_config, name, **kwargs
    ) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs

        self.model = GNNRLAgent(num_outputs)


    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        sim_state: Tensor = obs["ids_observation"].type(torch.FloatTensor)
        action_mask: Tensor = obs["action_mask"].type(torch.FloatTensor)
        edges = obs["edges"].type(torch.LongTensor)

        edges = edges.transpose(1, -1)
        sim_state = sim_state.unsqueeze(-1)

        policy_out, value_out = self.model(sim_state, edges, obs["defense_indices"].type(torch.LongTensor))

        self._value_out = value_out

        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return policy_out + inf_mask, state

    def value_function(self):
        return self._value_out.flatten()


def register_rllib_model():
    name = "GNNDefenderModel"
    ModelCatalog.register_custom_model(name, GNNDefenderModel)
    return name


register_rllib_model()
