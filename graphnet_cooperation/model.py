import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation

        # truncated-normal-like init (approximation)
        std = 1.0 / (in_dim ** 0.5)
        with torch.no_grad():
            self.linear.weight.normal_(0.0, std)
            self.linear.weight.clamp_(-2 * std, 2 * std)
            self.linear.bias.zero_()

    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = torch.tanh(x)
        return x


class GraphNetPlanner(nn.Module):
    """
    Simplified 2-step GraphNet-style planner.
    Output:
      - edge logits: [m, 2]
      - value: scalar
    """

    def __init__(self, node_dim=3, edge_dim=1, global_dim=2, hidden_dim=128):
        super().__init__()

        # Block 1
        self.phi1_e = MLP(edge_dim + 2 * node_dim + global_dim, hidden_dim, activation=True)
        self.phi1_v = MLP(hidden_dim + node_dim + global_dim, hidden_dim, activation=True)
        self.phi1_u = MLP(hidden_dim + hidden_dim + global_dim, hidden_dim, activation=True)

        # Block 2
        self.phi2_e = MLP(hidden_dim + 2 * hidden_dim + hidden_dim, 2, activation=False)
        self.phi2_v = MLP(hidden_dim + hidden_dim, hidden_dim, activation=True)
        self.phi2_u = MLP(2 + hidden_dim + hidden_dim, 1, activation=False)

    def forward(self, node_features, edge_features, global_features, edge_pairs, n_nodes):
        """
        node_features: [n, node_dim]
        edge_features: [m, edge_dim]
        global_features: [global_dim]
        edge_pairs: list[(i,j)]
        """

        device = node_features.device
        m = edge_features.shape[0]

        # ----- Block 1 edge update -----
        e1_list = []
        for k, (i, j) in enumerate(edge_pairs):
            x = torch.cat(
                [edge_features[k], node_features[i], node_features[j], global_features],
                dim=0,
            )
            e1_list.append(self.phi1_e(x))
        e1 = torch.stack(e1_list, dim=0)  # [m, 128]

        # aggregate edge messages to nodes
        agg1 = torch.zeros((n_nodes, e1.shape[1]), device=device)
        counts = torch.zeros((n_nodes, 1), device=device)
        for k, (i, j) in enumerate(edge_pairs):
            agg1[i] += e1[k]
            agg1[j] += e1[k]
            counts[i] += 1
            counts[j] += 1
        agg1 = agg1 / counts.clamp(min=1.0)

        # ----- Block 1 node update -----
        v1_list = []
        for i in range(n_nodes):
            x = torch.cat([agg1[i], node_features[i], global_features], dim=0)
            v1_list.append(self.phi1_v(x))
        v1 = torch.stack(v1_list, dim=0)  # [n, 128]

        # ----- Block 1 global update -----
        e1_sum = e1.mean(dim=0)
        v1_sum = v1.mean(dim=0)
        u1 = self.phi1_u(torch.cat([e1_sum, v1_sum, global_features], dim=0))  # [128]

        # ----- Block 2 edge update -> policy logits -----
        e2_list = []
        for k, (i, j) in enumerate(edge_pairs):
            x = torch.cat([e1[k], v1[i], v1[j], u1], dim=0)
            e2_list.append(self.phi2_e(x))
        edge_logits = torch.stack(e2_list, dim=0)  # [m, 2]

        # ----- Block 2 node update -----
        v2_list = []
        for i in range(n_nodes):
            x = torch.cat([v1[i], u1], dim=0)
            v2_list.append(self.phi2_v(x))
        v2 = torch.stack(v2_list, dim=0)

        # ----- Block 2 global update -> value -----
        e2_mean = edge_logits.mean(dim=0)
        v2_mean = v2.mean(dim=0)
        value = self.phi2_u(torch.cat([e2_mean, v2_mean, u1], dim=0)).squeeze(-1)

        return edge_logits, value