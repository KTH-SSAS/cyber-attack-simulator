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

        fc_hidden = model_config["fcnet_hiddens"]
        layers = len(fc_hidden)
        hidden_size = fc_hidden[0]

        self.model = GNNRLAgent(1, layers, hidden_size)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        policy_out, value_out = self.model.compute_action(obs)

        self._value_out = value_out

        return policy_out, state

    def value_function(self):
        return self._value_out.flatten()


def register_rllib_model():
    name = "GNNDefenderModel"
    ModelCatalog.register_custom_model(name, GNNDefenderModel)
    return name


register_rllib_model()
