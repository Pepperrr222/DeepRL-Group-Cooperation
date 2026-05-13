# model/graph_net.py
import torch
import torch.nn as nn
from utils.init_weights import init_weights

class StandardGraphNetBlock(nn.Module):
    """
    第一步消息传递 (First Message Passing Step).
    遵循标准的 GraphNet 更新逻辑。
    对应论文 Table 4 中的 phi_e^1, phi_v^1, phi_u^1。
    """
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__()
        
        # 1. Edge update function phi_e^1
        # Inputs: edge(e), sender(v_s), receiver(v_r), global(u)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim + global_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 2. Node update function phi_v^1
        # Inputs: aggregated_edges(sum e'), node(v), global(u)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + node_dim + global_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 3. Global update function phi_u^1
        # Inputs: aggregated_edges(sum e'), aggregated_nodes(sum v'), global(u)
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + global_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 应用截断正态分布初始化
        self.apply(init_weights)

    def forward(self, v, e, u, mask=None):
        B, N, _ = v.shape
        
        # --- Step 1: Edge Update (phi_e^1) ---
        # 扩展维度以便拼接
        v_s = v.unsqueeze(2).expand(-1, -1, N, -1)     # Sender nodes
        v_r = v.unsqueeze(1).expand(-1, N, -1, -1)     # Receiver nodes
        u_expand = u.view(B, 1, 1, -1).expand(-1, N, N, -1) # Global attributes
        
        edge_input = torch.cat([e, v_s, v_r, u_expand], dim=-1)
        e_prime = self.edge_mlp(edge_input) # (B, N, N, hidden_dim)
        
        # 屏蔽自环，避免向自身传递边信息
        if mask is not None:
            e_prime = e_prime * mask.unsqueeze(-1)

        # --- Step 2: Node Update (phi_v^1) ---
        # 聚合：将所有指向节点 r 的边特征求和
        sum_e_prime = e_prime.sum(dim=1) # (B, N, hidden_dim)
        
        u_expand_node = u.unsqueeze(1).expand(-1, N, -1)
        
        node_input = torch.cat([sum_e_prime, v, u_expand_node], dim=-1)
        v_prime = self.node_mlp(node_input) # (B, N, hidden_dim)

        # --- Step 3: Global Update (phi_u^1) ---
        # 聚合：全图的边和节点求和
        sum_e_prime_glob = e_prime.sum(dim=(1, 2)) # (B, hidden_dim)
        sum_v_prime_glob = v_prime.sum(dim=1)      # (B, hidden_dim)
        
        global_input = torch.cat([sum_e_prime_glob, sum_v_prime_glob, u], dim=-1)
        u_prime = self.global_mlp(global_input) # (B, hidden_dim)

        return v_prime, e_prime, u_prime


class ModifiedGraphNetBlock(nn.Module):
    """
    第二步消息传递 (Second Message Passing Step).
    对应论文 Table 4 中的 phi_e^2, phi_v^2, phi_u^2。
    
    关键修改：
    1. Node Update (phi_v^2) 不包含更新后的 Edge 特征。
    2. Edge Update 直接输出 Policy Logits，无激活函数。
    3. Global Update 直接输出 Value Estimate，无激活函数。
    """
    def __init__(self, input_dim, hidden_dim, action_dim=2):
        super().__init__()
        
        # 1. Edge Update (phi_e^2) -> 输出 Policy Logits
        # 论文 Table 4：无激活函数 (-)
        self.edge_mlp = nn.Linear(input_dim + 2 * input_dim + input_dim, action_dim)
        
        # 2. Node Update (phi_v^2) - MODIFIED
        # 论文高亮：Inputs do not include updated edge attributes.
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim + input_dim, hidden_dim), 
            nn.Tanh()
        )
        
        # 3. Global Update (phi_u^2) -> 输出 Value Estimate
        # 论文 Table 4：无激活函数 (-)
        # 注意：这里的输入包含了 Edge Logits (action_dim)
        self.global_mlp = nn.Linear(action_dim + hidden_dim + input_dim, 1)

        # 应用截断正态分布初始化
        self.apply(init_weights)

    def forward(self, v, e, u, mask=None):
        B, N, _ = v.shape
        
        # --- Step 1: Edge Update -> Policy Logits (E') ---
        v_s = v.unsqueeze(2).expand(-1, -1, N, -1)
        v_r = v.unsqueeze(1).expand(-1, N, -1, -1)
        u_expand = u.view(B, 1, 1, -1).expand(-1, N, N, -1)
        
        edge_input = torch.cat([e, v_s, v_r, u_expand], dim=-1)
        e_logits = self.edge_mlp(edge_input) # (B, N, N, action_dim)
        
        if mask is not None:
            e_logits = e_logits * mask.unsqueeze(-1)

        # --- Step 2: Node Update -> Unused Node Reps (V') ---
        # 核心解耦：这里只拼接了 v 和 u，直接无视了边特征 e_logits
        u_expand_node = u.unsqueeze(1).expand(-1, N, -1)
        
        node_input = torch.cat([v, u_expand_node], dim=-1)
        v_prime = self.node_mlp(node_input) # (B, N, hidden_dim)

        # --- Step 3: Global Update -> Value Estimate (u') ---
        sum_e_logits = e_logits.sum(dim=(1, 2)) # (B, action_dim)
        sum_v_prime = v_prime.sum(dim=1)        # (B, hidden_dim)
        
        global_input = torch.cat([sum_e_logits, sum_v_prime, u], dim=-1)
        u_value = self.global_mlp(global_input) # (B, 1)

        return v_prime, e_logits, u_value