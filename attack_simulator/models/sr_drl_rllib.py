import torch

from torch.nn import Module, Sequential, Linear, LeakyReLU, Sigmoid
from torch_geometric.data import Data, Batch

from attack_simulator.models.sr_drl import MultiMessagePassing

class SRDRLAGENT(Module):
	
    def __init__(self) -> None:
        super().__init__()
        self.net = Net()

    def forward(self, obs):

        node_feats, edge_attr, edge_index = (obs["ids_observation"],
                                        None,
                                        obs["edges"])
        
        s_batch = (node_feats, edge_attr, edge_index)
        defender_action = self.net(s_batch)
        _, value, action_probs, node_probs = defender_action

        return torch.cat([action_probs, node_probs], dim=-1), value
    



EMB_SIZE = 8


class Net(Module):
    def __init__(
        self,
        multi=False,
        mp_iterations=1,
        device="cpu",
    ):
        super().__init__()

        self.embed_node = Sequential(Linear(1, EMB_SIZE), LeakyReLU())

        self.mp_main = MultiMessagePassing(steps=mp_iterations)

        if multi:
            self.a_0_sel = Sequential(
                Linear(EMB_SIZE, 1), Sigmoid()
            )  # Bernoulli trial for each node

        else:
            self.a_0_sel = Linear(EMB_SIZE, 2)  # select, no-op #TODO change to not be hardcoded
            self.a_1_sel = Linear(EMB_SIZE, 1)  # node features -> node probability

        self.value_function = Linear(EMB_SIZE, 1)  # global features -> state value
        self.device = torch.device(device)
        self.to(self.device)
        self.multi = multi

    def save(self, file="model.pt"):
        torch.save(self.state_dict(), file)

    def load(self, file="model.pt"):
        self.load_state_dict(torch.load(file, map_location=self.device))

    def copy_weights(self, other, rho):
        params_other = list(other.parameters())
        params_self = list(self.parameters())

        for i in enumerate(params_other):
            val_self = params_self[i].data
            val_other = params_other[i].data
            val_new = rho * val_other + (1 - rho) * val_self

            params_self[i].data.copy_(val_new)

    def forward(self, s_batch):
        
        node_feats, edge_attr, edge_index = s_batch
        edge_attr = [torch.empty(0, dtype=torch.float32, device=self.device) for _ in range(edge_index.shape[0])]
        edge_index = edge_index.type(torch.int64).to(self.device)

        # create batch
        def convert_batch(feats, edge_attr, edge_index):
            data = [
                Data(x=feats[i], edge_attr=edge_attr[i], edge_index=edge_index[i])
                for i in range(feats.shape[0])
            ]
            batch = Batch.from_data_list(data)
            batch_ind = batch.batch.to(self.device)

            return data, batch, batch_ind

        batch: Batch
        _, batch, batch_ind = convert_batch(node_feats, edge_attr, edge_index)

        # process state
        x = self.embed_node(batch.x.reshape(-1, 1))
        xg = torch.zeros(batch.num_graphs, EMB_SIZE, device=self.device)

        x, xg = self.mp_main(x, xg, batch.edge_attr, batch.edge_index, batch_ind, batch.num_graphs)

        # decode value
        value = self.value_function(xg)

        # ========== MULTI-SELECT ===========
        if self.multi:
            a0_activation = self.a_0_sel(x).flatten()
        
        # ========== SINGLE-SELECT ===========
        else:
            a0_activation = self.a_0_sel(xg)
            a1_activation = self.a_1_sel(x)

        return None, value, a0_activation, a1_activation.reshape(node_feats.shape[0], -1)  # todo, add noop prob
