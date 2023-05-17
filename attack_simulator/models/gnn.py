import einops
import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.nn.models import GIN
from torch_geometric.utils import add_self_loops, degree
from torch import LongTensor, FloatTensor

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

FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38

class GNNRLAgent(nn.Module):
    def __init__(self, channels_in, num_layers, hidden_channels):
        super().__init__()
        channels_in = 1
        channels_out = hidden_channels
        num_layers = num_layers
        # self.embedding_func = GCNConv(1, channels_out)
        self.embedding_func = GIN(channels_in, channels_out, num_layers)
        self.activation = nn.ReLU()
        self.pool = TopKPooling(channels_out, ratio=0.8)

        self.policy_fn = nn.Linear(channels_out, 1)
        self.value_fn = nn.Linear(channels_out, 1)

    def compute_action(self, obs):
        sim_state: FloatTensor = obs["ids_observation"].type(torch.FloatTensor)
        action_mask: FloatTensor = obs["action_mask"].type(torch.FloatTensor)
        edges: LongTensor = obs["edges"].type(LongTensor)
        defense_indices: LongTensor = obs["defense_indices"].type(LongTensor)

        defense_indices = defense_indices.unsqueeze(0) if len(defense_indices.shape) == 1 else defense_indices 

        sim_state = sim_state.unsqueeze(-1)

        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        batch: Batch = batch_to_gnn_batch(sim_state, edges)

        action_dist, value_pred = self.forward(batch, defense_indices, obs["nop_index"])

        action_dist = action_dist + inf_mask

        return action_dist, value_pred
    

    def forward(self, batch: Batch, defense_indices: LongTensor, nop_index: int):
        # x has shape [B, N, in_channels]
        # N is the number of nodes
        # B is the batch size
        # in_channels is the number of features per node
        # edge_index has shape [B, 2, E]
        # B is the batch size
        # E is the number of edges

        # convert defense indices to batched indices
        # for i in range(batch_size):
        #    defense_indices[i] += i * num_nodes

        # defense_edges, _ = subgraph(defense_indices, batch.edge_index)

        # embedded graph has shape [BxN, channels_out]
        embedded_graph = self.activation(self.embedding_func(batch.x, batch.edge_index))

        # Reshape embedding to [B, N, channels_out]
        # This assumes that all graphs in the batch have the same number of nodes
        # https://einops.rocks/1-einops-basics/#decomposition-of-axis
        embedding = einops.rearrange(embedded_graph, "(b n) c -> b n c", b=batch.num_graphs)

        # gather embeddings for defense step
        defense_indices = einops.repeat(defense_indices, "b k -> b k d", d=embedding.shape[-1])
        defense_embeddings = torch.gather(embedding, 1, defense_indices)

        # Compute policy and value outputs
        # policy_out has shape [B, num_outputs]
        # only compute policy func for defense nodes
        wait_embeddings = defense_embeddings[:, nop_index, :]

        # pool_out = self.pool(embedded_graph, defense_edges)
        policy_out = self.policy_fn(defense_embeddings).squeeze()

        # Take the mean of the node embeddings to get a (rough) graph embedding
        # value_out has shape [B, 1]
        value_out = self.value_fn(wait_embeddings).squeeze()

        return policy_out, value_out


def batch_to_gnn_batch(x, edge_index):
    # x has shape [B, N, in_channels]
    # N is the number of nodes
    # B is the batch size
    # in_channels is the number of features per node
    # edge_index has shape [B, 2, E]
    # B is the batch size
    # E is the number of edges
    x = x.unsqueeze(0) if len(x.shape) == 2 else x
    edge_index = edge_index.unsqueeze(0) if len(edge_index.shape) == 2 else edge_index
    batch_size = x.shape[0]
    return Batch.from_data_list([Data(x=x[i], edge_index=edge_index[i]) for i in range(batch_size)])
