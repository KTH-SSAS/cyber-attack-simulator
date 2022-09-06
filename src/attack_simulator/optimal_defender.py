from email import policy
from sre_constants import BIGCHARSET
from typing import Dict, List
import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
from torch import Tensor
import numpy as np
from torch import nn
from attack_simulator.graph import AttackGraph

BIG_SMALL_NUMBER = -1e-30

def register_rllib_model():
    name = "OptimalDefenderModel"
    ModelCatalog.register_custom_model(name, OptimalDefenderModel)
    return name


class OptimalDefenderModel(TorchModelV2, nn.Module):
    """Policy for the agent agent."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.num_outputs = num_outputs
        self.compromised_steps = torch.zeros(1, 1)
        self.attack_steps_by_defense_steps: List[List[int]] = kwargs["defense_steps"]
        self.num_defense_steps = len(self.attack_steps_by_defense_steps)
        self.value_fn = nn.Linear(obs_space.original_space.spaces["sim_obs"].shape[0], 1)
        self.value_out = torch.ones(1, 1)

    def forward(self, input_dict, state, seq_lens):

        obs = input_dict["obs"]
        sim_state: Tensor = obs["sim_obs"].type(torch.FloatTensor)

        attack_state = sim_state[:, self.num_defense_steps:]

        self.value_out = self.value_fn(sim_state)

        if attack_state.shape != self.compromised_steps.shape:
            self.compromised_steps = torch.zeros_like(attack_state)

        new_compromised_steps = attack_state - self.compromised_steps
        indeces = torch.nonzero(new_compromised_steps, as_tuple=False)
        self.compromised_steps += new_compromised_steps

        defenses_to_activate = torch.ones(attack_state.shape[0], self.num_outputs)
        defenses_to_activate *= FLOAT_MIN

        # Nothing has been attacked yet
        if torch.sum(attack_state).item() == 0:
            defenses_to_activate[0] = 1
            return defenses_to_activate, state

    
        # Check if a defense is triggered
        for b, a_s in indeces:
            for d_s, defendable_steps in enumerate(self.attack_steps_by_defense_steps):
                if a_s.item() in defendable_steps:
                    defenses_to_activate[b, d_s+1] = 1

                
        # If there is no defense to acivate, wait.
        for b, x in enumerate(defenses_to_activate):
            if torch.max(x).item() < FLOAT_MIN + 1000.0:
                defenses_to_activate[b][0] = 1

        action_mask: Tensor = obs["action_mask"].type(torch.FloatTensor)

        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        policy_out = defenses_to_activate + inf_mask

        policy_out = torch.clamp(policy_out, FLOAT_MIN, FLOAT_MAX)

        return policy_out, state

    def value_function(self):
        return self.value_out.flatten()

    def import_from_h5(self, h5_file: str) -> None:
        return super().import_from_h5(h5_file)
