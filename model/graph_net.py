import torch
import torch.nn as nn

class MLP(nn.Module):
    """辅助类：简单的多层感知机 (Multi-Layer Perceptron)，用作更新函数 phi"""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), # 论文通常使用 Tanh 或 ReLU
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # 保持输出在激活空间，或者最后一层去掉激活
        )
        # 注意：如果是输出层(logits/value)，最后一层通常没有激活函数。
        # 这里为了通用性，在下面的 Block 中如果是输出层会重写。

    def forward(self, x):
        return self.net(x)

class StandardGraphNetBlock(nn.Module):
    """
    对应的文中描述的第一步消息传递 (First Message Passing Step).
    遵循标准的 GraphNet 更新逻辑。
    """
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__()
        
        # 1. Edge update function phi_e
        # Inputs: edge(e), sender(v_s), receiver(v_r), global(u)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim + global_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 2. Node update function phi_v
        # Inputs: aggregated_edges(sum e'), node(v), global(u)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + node_dim + global_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 3. Global update function phi_u
        # Inputs: aggregated_edges(sum e'), aggregated_nodes(sum v'), global(u)
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + global_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, v, e, u, mask=None):
        """
        v: Node attributes (Batch, N, Dv)
        e: Edge attributes (Batch, N, N, De)
        u: Global attributes (Batch, Du)
        mask: Adjacency mask (Batch, N, N) 用于屏蔽对角线自环
        """
        B, N, _ = v.shape
        
        # --- Step 1: Edge Update (phi_e) ---
        # e'_sr = phi_e(e_sr, v_s, v_r, u)
        v_s = v.unsqueeze(2).expand(-1, -1, N, -1)     # Sender nodes
        v_r = v.unsqueeze(1).expand(-1, N, -1, -1)     # Receiver nodes
        u_expand = u.view(B, 1, 1, -1).expand(-1, N, N, -1) # Global expanded to edges
        
        # Concatenate inputs: [e, v_s, v_r, u]
        edge_input = torch.cat([e, v_s, v_r, u_expand], dim=-1)
        e_prime = self.edge_mlp(edge_input)
        
        if mask is not None:
            e_prime = e_prime * mask.unsqueeze(-1)

        # --- Step 2: Node Update (phi_v) ---
        # v'_r = phi_v(sum_s(e'_sr), v_r, u)
        
        # Aggregation: Sum edge attributes for each receiver node r
        sum_e_prime = e_prime.sum(dim=1) # (Batch, N, hidden_dim)
        
        u_expand_node = u.unsqueeze(1).expand(-1, N, -1) # Global expanded to nodes
        
        # Concatenate inputs: [sum_e', v, u]
        node_input = torch.cat([sum_e_prime, v, u_expand_node], dim=-1)
        v_prime = self.node_mlp(node_input)

        # --- Step 3: Global Update (phi_u) ---
        # u' = phi_u(sum_sr(e'_sr), sum_r(v'_r), u)
        
        # Aggregation
        sum_e_prime_glob = e_prime.sum(dim=(1, 2)) # Sum over all edges
        sum_v_prime_glob = v_prime.sum(dim=1)      # Sum over all nodes
        
        # Concatenate inputs: [sum_e', sum_v', u]
        global_input = torch.cat([sum_e_prime_glob, sum_v_prime_glob, u], dim=-1)
        u_prime = self.global_mlp(global_input)

        return v_prime, e_prime, u_prime


class ModifiedGraphNetBlock(nn.Module):
    """
    对应文中描述的第二步消息传递 (Second Message Passing Step).
    
    关键修改点 (文中黄色高亮):
    "The second node-update function, phi^2_v, is modified so that its inputs 
     do not include updated edge attributes."
    """
    def __init__(self, input_dim, hidden_dim, action_dim=2):
        super().__init__()
        
        # 1. Edge Update (phi^2_e)
        # Output: Policy Logits (E') -> Dimension = action_dim
        # 输入维度是上一层的 hidden_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim + 2 * input_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim) # 输出 Policy Logits
        )
        
        # 2. Node Update (phi^2_v) - MODIFIED
        # Inputs: node(v), global(u) -> NO EDGE INPUTS
        # Output: Unused node representations (V')
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim + input_dim, hidden_dim), # Input dim reduced!
            nn.Tanh()
        )
        
        # 3. Global Update (phi^2_u)
        # Inputs: aggregated_edges(sum e'), aggregated_nodes(sum v'), global(u)
        # Output: Value estimate (u') -> Dimension = 1
        self.global_mlp = nn.Sequential(
            nn.Linear(action_dim + hidden_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # 输出 Value Estimate
        )

    def forward(self, v, e, u, mask=None):
        B, N, _ = v.shape
        
        # --- Edge Update ---
        # Same logic, but outputs logits E'
        v_s = v.unsqueeze(2).expand(-1, -1, N, -1)
        v_r = v.unsqueeze(1).expand(-1, N, -1, -1)
        u_expand = u.view(B, 1, 1, -1).expand(-1, N, N, -1)
        
        edge_input = torch.cat([e, v_s, v_r, u_expand], dim=-1)
        e_logits = self.edge_mlp(edge_input) # E' (Policy Logits)
        
        if mask is not None:
            e_logits = e_logits * mask.unsqueeze(-1)

        # --- Node Update (MODIFIED) ---
        # v'_r = phi_v(v_r, u)  <-- 注意这里去掉了 sum_e_prime
        
        u_expand_node = u.unsqueeze(1).expand(-1, N, -1)
        
        # Concatenate inputs: [v, u] ONLY
        node_input = torch.cat([v, u_expand_node], dim=-1)
        v_prime = self.node_mlp(node_input) # V' (Unused representations)

        # --- Global Update ---
        # u' = phi_u(sum(E'), sum(V'), u)
        # Note: Uses E' (logits) and V' for value estimation
        sum_e_logits = e_logits.sum(dim=(1, 2))
        sum_v_prime = v_prime.sum(dim=1)
        
        global_input = torch.cat([sum_e_logits, sum_v_prime, u], dim=-1)
        u_value = self.global_mlp(global_input) # u' (Value Estimate)

        return v_prime, e_logits, u_value


class GraphNetAgent(nn.Module):
    """
    Social Planner Agent Architecture.
    Sequences two GraphNet modules.
    """
    def __init__(self, node_in_dim, edge_in_dim, global_in_dim, hidden_dim=128):
        super().__init__()
        
        # First Message Passing Step (Standard)
        self.block1 = StandardGraphNetBlock(
            node_dim=node_in_dim,
            edge_dim=edge_in_dim,
            global_dim=global_in_dim,
            hidden_dim=hidden_dim
        )
        
        # Second Message Passing Step (Modified)
        # Inputs to this block are the hidden outputs of Block 1
        self.block2 = ModifiedGraphNetBlock(
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim,
            action_dim=2 # e.g., Logits for [Keep, Change]
        )

    def forward(self, v, e, u, mask=None):
        # 1. First GraphNet Module
        # G' = Block1(G)
        v_hidden, e_hidden, u_hidden = self.block1(v, e, u, mask)
        
        # 2. Second GraphNet Module
        # "Output... value estimate u', unused node V', and policy logits E'"
        _, policy_logits, value_estimate = self.block2(v_hidden, e_hidden, u_hidden, mask)
        
        return policy_logits, value_estimate