import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data, Batch
import einops


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(1))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class GNNRLAgent(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        channels_out = 8
        self.embedding_func = GCNConv(1, channels_out)

        self.policy_fn = nn.Linear(channels_out, 1)
        self.value_fn = nn.Linear(channels_out, 1)

        self.wait_embedding = nn.Embedding(1, channels_out)
        self.terminate_embedding = nn.Embedding(1, channels_out)

    def forward(self, x, edge_index, defense_indices):
        # x has shape [B, N, in_channels]
        # N is the number of nodes
        # B is the batch size
        # in_channels is the number of features per node
        # edge_index has shape [B, 2, E]
        # B is the batch size
        # E is the number of edges
        batch = construct_gnn_batch(x, edge_index)

        # embedding has shape [BxN, channels_out]
        embedding = self.embedding_func(batch.x, batch.edge_index)

        # Reshape embedding to [B, N, channels_out]
        # This assumes that all graphs in the batch have the same number of nodes
        # https://einops.rocks/1-einops-basics/#decomposition-of-axis
        embedding = einops.rearrange(embedding, "(b n) c -> b n c", b=batch.num_graphs)

        # gather embeddings for defense step
        defense_indices = einops.repeat(defense_indices, "b k -> b k d", d=embedding.shape[-1])
        defense_embeddings = torch.gather(embedding, 1, defense_indices)

        # add embeddings for wait and terminate actions
        wait_repeated = einops.repeat(
            self.wait_embedding.weight, "k d -> b k d", b=embedding.shape[0]
        )
        terminate_repeated = einops.repeat(
            self.terminate_embedding.weight, "k d -> b k d", b=embedding.shape[0]
        )
        policy_in = torch.cat([defense_embeddings, wait_repeated, terminate_repeated], dim=1)

        # Compute policy and value outputs
        # policy_out has shape [B, num_outputs]
        # only compute policy func for defense nodes
        policy_out = self.policy_fn(policy_in).squeeze()

        # Take the mean of the node embeddings to get a (rough) graph embedding
        # value_out has shape [B, 1]
        value_out = self.value_fn(embedding.mean(dim=1)).squeeze()

        return policy_out, value_out


def construct_gnn_batch(x, edge_index):
    # x has shape [B, N, in_channels]
    # N is the number of nodes
    # B is the batch size
    # in_channels is the number of features per node
    # edge_index has shape [B, 2, E]
    # B is the batch size
    # E is the number of edges
    batch_size = x.shape[0]
    return Batch.from_data_list([Data(x=x[i], edge_index=edge_index[i]) for i in range(batch_size)])
