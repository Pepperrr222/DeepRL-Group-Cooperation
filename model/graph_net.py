# model/graph_net.py
import torch
import torch.nn as nn
from utils.init_weights import init_weights

class StandardGraphNetBlock(nn.Module):
    """
    对应论文中的第一步 Message Passing Step (Phi^1).
    这是一个标准的 GraphNet 模块。
    """
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__()
        
        # 1. Edge Update function (phi^1_e)
        # Inputs: edge_attr, sender_node, receiver_node, global_attr
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim + global_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 2. Node Update function (phi^1_v)
        # Inputs: aggregated_edges, node_attr, global_attr
        # 这里的 inputs 包含 summed edges
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + node_dim + global_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 3. Global Update function (phi^1_u)
        # Inputs: aggregated_edges, aggregated_nodes, global_attr
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + global_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.apply(init_weights)

    def forward(self, v, e, u, mask):
        """
        v: Nodes (B, N, Dv)
        e: Edges (B, N, N, De)
        u: Globals (B, Du)
        mask: Adjacency mask (B, N, N) - 0 on diagonal
        """
        B, N, _ = v.shape
        
        # --- Edge Update (phi^1_e) ---
        # e'_sr = phi_e(e_sr, v_s, v_r, u)
        v_s = v.unsqueeze(2).expand(-1, -1, N, -1)     # Sender
        v_r = v.unsqueeze(1).expand(-1, N, -1, -1)     # Receiver
        u_expand = u.view(B, 1, 1, -1).expand(-1, N, N, -1)
        
        edge_input = torch.cat([e, v_s, v_r, u_expand], dim=-1)
        e_prime = self.edge_mlp(edge_input) # (B, N, N, H)
        
        # Masking self-loops for aggregation
        e_prime_masked = e_prime * mask.unsqueeze(-1)

        # --- Node Update (phi^1_v) ---
        # v'_r = phi_v(sum_s(e'_sr), v_r, u)
        sum_e_prime = e_prime_masked.sum(dim=1) # Sum over senders -> (B, N, H)
        u_expand_node = u.unsqueeze(1).expand(-1, N, -1)
        
        node_input = torch.cat([sum_e_prime, v, u_expand_node], dim=-1)
        v_prime = self.node_mlp(node_input) # (B, N, H)

        # --- Global Update (phi^1_u) ---
        # u' = phi_u(sum_sr(e'_sr), sum_r(v'_r), u)
        sum_e_prime_glob = e_prime_masked.sum(dim=(1, 2)) # (B, H)
        sum_v_prime_glob = v_prime.sum(dim=1)             # (B, H)
        
        global_input = torch.cat([sum_e_prime_glob, sum_v_prime_glob, u], dim=-1)
        u_prime = self.global_mlp(global_input) # (B, H)

        return v_prime, e_prime, u_prime


class ModifiedGraphNetBlock(nn.Module):
    """
    对应论文中的第二步 Message Passing Step (Phi^2).
    关键修改: "The second node-update function phi^2_v is modified 
    so that its inputs do not include updated edge attributes."
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # Phi^2_e: Edge Update
        # Output is POLICY LOGITS (2 dimensions)
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim + 2 * input_dim + input_dim, 2), # Output dim 2
            # No activation here usually for logits, but if following purely MLP pattern:
            # Paper says "producing policy logits E'". Usually final layer is linear.
            # Assuming linear output for logits.
        )
        
        # Phi^2_v: Node Update (MODIFIED)
        # Inputs: node_attr, global_attr (NO EDGE INPUT)
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim + input_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Phi^2_u: Global Update
        # Inputs: aggregated_edges (logits), aggregated_nodes, global_attr
        # Output is VALUE ESTIMATE (1 dimension)
        self.global_mlp = nn.Sequential(
            nn.Linear(2 + hidden_dim + input_dim, 1), # 2 comes from edge logits
             # Linear output for value
        )
        
        self.apply(init_weights)

    def forward(self, v, e, u, mask):
        """
        Inputs come from Block 1 (all are hidden_dim size)
        """
        B, N, _ = v.shape
        
        # --- Edge Update (phi^2_e) ---
        # Outputs Policy Logits directly
        v_s = v.unsqueeze(2).expand(-1, -1, N, -1)
        v_r = v.unsqueeze(1).expand(-1, N, -1, -1)
        u_expand = u.view(B, 1, 1, -1).expand(-1, N, N, -1)
        
        edge_input = torch.cat([e, v_s, v_r, u_expand], dim=-1)
        e_logits = self.edge_mlp(edge_input) # (B, N, N, 2)
        
        e_logits_masked = e_logits * mask.unsqueeze(-1)

        # --- Node Update (phi^2_v) ---
        # *** MODIFIED: Inputs do not include updated edge attributes ***
        # v''_r = phi_v(v'_r, u')
        u_expand_node = u.unsqueeze(1).expand(-1, N, -1)
        
        # Only concatenating Node and Global
        node_input = torch.cat([v, u_expand_node], dim=-1) 
        v_prime = self.node_mlp(node_input) # (B, N, H)

        # --- Global Update (phi^2_u) ---
        # u'' = phi_u(sum(e''), sum(v''), u')
        # e'' here are the logits (dim 2)
        sum_e_logits = e_logits_masked.sum(dim=(1, 2)) # (B, 2)
        sum_v_prime = v_prime.sum(dim=1)               # (B, H)
        
        global_input = torch.cat([sum_e_logits, sum_v_prime, u], dim=-1)
        u_value = self.global_mlp(global_input) # (B, 1)

        return e_logits, u_value