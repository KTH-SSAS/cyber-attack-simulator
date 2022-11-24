from itertools import repeat
import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
from torch import Tensor, nn
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
from ray.rllib.algorithms.algorithm import Algorithm
import numpy as np

from attack_simulator.reward_utils import action_cost_for_timestep, get_minimum_rewards, normalize

def register_rllib_model():
    name = "DefenderModel"
    ModelCatalog.register_custom_model(name, DefenderModel)
    name = "DQNDefenderModel"
    ModelCatalog.register_custom_model(name, DQNDefenderModel)
    return name

class Defender(PPO):
    _allow_unknown_configs = True
    def get_default_policy_class(self, config):
        return DefenderPolicy

class DefenderPolicy(PPOTorchPolicy):

    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        with torch.no_grad():
            if self.config["scale_rewards"]:
                
                    #sample_batch["rewards"] = rewards / len(rewards)

                    try:
                        rewards = sample_batch["rewards"]
                        info = sample_batch["infos"][-1]
                        avg_defense_cost = np.mean(info["defense_costs"])
                        num_defenses = len(info["defense_costs"])
                        avg_flag_cost = np.mean(info["flag_costs"])
                        episode_length = len(rewards)
                        min_defense_rewards = get_minimum_rewards(avg_defense_cost, num_defenses, episode_length)
                        min_flag_rewards = repeat(avg_flag_cost, episode_length)
                        total_max_reward_per_timestep = map(sum, zip(min_defense_rewards, min_flag_rewards))
                        scaled_rewards = map(lambda x: normalize(x[0], -x[1], 0, -1, 1), zip(rewards, total_max_reward_per_timestep))
                        sample_batch["rewards"] = np.array(list(scaled_rewards)) / episode_length

                    except IndexError:
                        pass
                
                # rewards = sample_batch["rewards"]


                # postprocessed_rewards = [normalize(reward, min_d+min_a, 0, 0, 1) for reward, min_d, min_a in zip(rewards, min_defense_rewards, min_flag_rewards)]

                # sample_batch["rewards"] = postprocessed_rewards


            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )

class HiddenLayer(nn.Module):

    def __init__(self, in_features, out_features, activation_func) -> None:
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.activation_func = activation_func

    def forward(self, x):
        return self.activation_func(self.layer(x))

class DefenderModel(TorchModelV2, nn.Module):
    """Policy for the agent agent."""

    _value_out: Tensor

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_space = obs_space
        sim_space = obs_space.original_space.spaces["sim_obs"]
        self.action_space = action_space
        self.num_outputs = num_outputs
        activation_func = nn.Tanh
        self.vf_share_layers = model_config["vf_share_layers"]

        hidden_layers = []
        prev_layer_size = sim_space.shape[0]
        for dim in model_config['fcnet_hiddens']:
            hidden_layers.append(HiddenLayer(prev_layer_size, dim, activation_func()))
            prev_layer_size = dim

        self.embedding_func = nn.Sequential(
            *hidden_layers,
        )

        if not self.vf_share_layers:
            prev_layer_size = sim_space.shape[0]
            hidden_layers = []
            for dim in model_config['fcnet_hiddens']:
                hidden_layers.append(HiddenLayer(prev_layer_size, dim, activation_func()))
                prev_layer_size = dim
            self.vf_embedding_func = nn.Sequential(
                *hidden_layers,
            )

        self.policy_fn = nn.Linear(prev_layer_size, num_outputs)
        self.value_fn = nn.Linear(prev_layer_size, 1)
        self.use_cuda = torch.cuda.is_available()
        
    def forward(self, input_dict, state, seq_lens):

        obs = input_dict["obs"]

        sim_state: Tensor = obs["sim_obs"].type(torch.FloatTensor)
        action_mask: Tensor = obs["action_mask"].type(torch.FloatTensor)

        if self.use_cuda:
            sim_state = sim_state.cuda()
            action_mask = action_mask.cuda()

        try:
            embedding = self.embedding_func(sim_state)
        except RuntimeError:
            self.use_cuda = False
            sim_state = sim_state.cpu()
            action_mask = action_mask.cpu()
            embedding = self.embedding_func(sim_state)

        policy_out = self.policy_fn(embedding)
        if self.vf_share_layers:
            value_out = self.value_fn(embedding)
        else:
            value_out = self.value_fn(self.vf_embedding_func(sim_state))
        self._value_out = value_out

        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return policy_out + inf_mask, state

    def value_function(self):
        return self._value_out.flatten()

    def import_from_h5(self, h5_file: str) -> None:
        return super().import_from_h5(h5_file)


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
        **kw
    ):
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )

        self.obs_space = obs_space
        sim_space = obs_space.original_space.spaces["sim_obs"]
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

        sim_state: Tensor = obs["sim_obs"].type(torch.FloatTensor)
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