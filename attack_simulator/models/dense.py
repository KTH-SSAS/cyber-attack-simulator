from torch import nn
import torch


class HiddenLayer(nn.Module):
    def __init__(self, in_features, out_features, activation_func) -> None:
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.activation_func = activation_func

    def forward(self, x):
        return self.activation_func(self.layer(x))


class MLP(nn.Module):
    def __init__(self, input_size, hidden_dims) -> None:
        super().__init__()
        activation_func = nn.Tanh
        hidden_layers = []
        prev_layer_size = input_size
        self.output_dim = hidden_dims[-1]
        for dim in hidden_dims:
            hidden_layers.append(HiddenLayer(prev_layer_size, dim, activation_func()))
            prev_layer_size = dim

        self.mlp = nn.Sequential(
            *hidden_layers,
        )

    def forward(self, x):
        return self.mlp(x)


class MLPRLAgent(nn.Module):
    def __init__(self, input_size, mlp_hidden_dims, policy_output_size, vf_share_layers) -> None:
        super().__init__()
        self.embedding_func = MLP(input_size, mlp_hidden_dims)

        if not vf_share_layers:
            self.vf_embedding_func = MLP(input_size, mlp_hidden_dims)
        else:
            self.vf_embedding_func = self.embedding_func

        self.policy_fn = nn.Linear(mlp_hidden_dims[-1], policy_output_size)
        self.value_fn = nn.Linear(mlp_hidden_dims[-1], 1)
        self.vf_share_layers = vf_share_layers

    def forward(self, x):
        embedding = self.embedding_func(x)
        policy_out = self.policy_fn(embedding)

        if self.vf_share_layers:
            value_out = self.value_fn(embedding)
        else:
            value_out = self.value_fn(self.vf_embedding_func(x))

        return policy_out, value_out

class HierarchicalMLPRLAgent(nn.Module):
    def __init__(self, input_size, mlp_hidden_dims, num_a0_actions, num_a1_actions, vf_share_layers) -> None:
        super().__init__()
        self.embedding_func = MLP(input_size, mlp_hidden_dims)

        if not vf_share_layers:
            self.vf_embedding_func = MLP(input_size, mlp_hidden_dims)
        else:
            self.vf_embedding_func = self.embedding_func

        self.a0_policy_fn = nn.Linear(mlp_hidden_dims[-1], num_a0_actions)
        self.a1_policy_fn = nn.Linear(mlp_hidden_dims[-1] + num_a0_actions, num_a1_actions)
        self.value_fn = nn.Linear(mlp_hidden_dims[-1], 1)
        self.vf_share_layers = vf_share_layers

    def forward(self, x):
        embedding = self.embedding_func(x)
        a0_out = self.a0_policy_fn(embedding)
        a1_out = self.a1_policy_fn(torch.cat([embedding, a0_out], dim=-1))

        if self.vf_share_layers:
            value_out = self.value_fn(embedding)
        else:
            value_out = self.value_fn(self.vf_embedding_func(x))

        return torch.cat([a0_out, a1_out], dim=-1), value_out