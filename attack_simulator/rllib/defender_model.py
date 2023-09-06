from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium.spaces import Box
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
from torch import Tensor

from ..models.dense import MLPRLAgent


class DefenderModel(TorchModelV2, nn.Module):
    """Policy for the agent agent."""

    _value_out: Tensor

    def __init__(
        self, obs_space: Box, action_space, num_outputs, model_config, name, **kwargs
    ) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_space = obs_space
        sim_space = obs_space.original_space.spaces["ids_observation"]
        self.action_space = action_space
        self.num_outputs = num_outputs

        input_size = sim_space.shape[0]
        self.model = MLPRLAgent(
            input_size, model_config["fcnet_hiddens"], num_outputs, model_config["vf_share_layers"]
        )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        sim_state: Tensor = obs["ids_observation"].type(torch.FloatTensor)
        action_mask: Tensor = obs["action_mask"].type(torch.FloatTensor)

        policy_out, value_out = self.model(sim_state)

        self._value_out = value_out

        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return policy_out + inf_mask, state

    def value_function(self):
        return self._value_out.flatten()

    def import_from_h5(self, h5_file: str) -> None:
        return


class DQNDefenderModel(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        true_obs_shape=(4,),
        action_embed_size=2,
        **kw,
    ):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kw)

        self.obs_space = obs_space
        sim_space = obs_space.original_space.spaces["ids_observation"]
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
        self.use_cuda = torch.cuda.is_available()

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        sim_state: Tensor = obs["ids_observation"].type(torch.FloatTensor)
        action_mask: Tensor = obs["action_mask"].type(torch.FloatTensor)

        if action_mask.sum().item() == 0:
            action_mask = torch.ones_like(action_mask)

        if self.use_cuda:
            sim_state = sim_state.cuda()
            action_mask = action_mask.cuda()

        # Compute the predicted action embedding
        try:
            embed = self.embedding_func(sim_state)
        except RuntimeError:
            self.use_cuda = False
            sim_state = sim_state.cpu()
            action_mask = action_mask.cpu()
            embed = self.embedding_func(sim_state)

        policy_out = self.policy_fn(embed)

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), -1e10, 1e10)

        return policy_out + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()

    def import_from_h5(self, h5_file: str) -> None:
        return


def register_rllib_model():
    name = "DefenderModel"
    ModelCatalog.register_custom_model(name, DefenderModel)
    name = "DQNDefenderModel"
    ModelCatalog.register_custom_model(name, DQNDefenderModel)
    return name


register_rllib_model()
