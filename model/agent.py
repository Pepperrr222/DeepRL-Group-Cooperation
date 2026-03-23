# model/agent.py
import torch
import torch.nn as nn
from model.graph_net import StandardGraphNetBlock, ModifiedGraphNetBlock
from config import ModelConfig, GameConfig, MODE
from utils.init_weights import init_weights

class SocialPlannerAgent_v1(nn.Module):
    def __init__(self):
        super().__init__()
        
        h_dim = ModelConfig.HIDDEN_DIM # Usually 128
        
        # --- Block 1 (Standard) ---
        # 这一层负责特征提取：将低维的 (2, 1, 1) 映射到高维 (128)
        self.block1 = StandardGraphNetBlock(
            node_dim=ModelConfig.NODE_IN_DIM,   # 2
            edge_dim=ModelConfig.EDGE_IN_DIM,   # 1
            global_dim=ModelConfig.GLOBAL_IN_DIM, # 1
            hidden_dim=h_dim
        )
        
        # --- Block 2 (Modified) ---
        # 这一层负责决策：输入是 128 维的隐向量
        # 输出是 Policy (2维 Logits) 和 Value (1维 Scalar)
        self.block2 = ModifiedGraphNetBlock(
            input_dim=h_dim,
            hidden_dim=h_dim,
            action_dim=2  # Explicitly define output dimension for policy logits
        )
        
        self.apply(init_weights)

    def forward(self, capital, prev_decisions, adj_matrix, time_step):
        """
        Args:
            capital: (B, N)
            prev_decisions: (B, N)
            adj_matrix: (B, N, N)
            time_step: int or Tensor (scalar)
        """
        B, N = capital.shape
        device = capital.device
        
        # 1. Feature Construction (G = u, V, E)
        # 确保输入为 float 类型，防止类型不匹配错误
        v = torch.stack([capital.float(), prev_decisions.float()], dim=-1) # (B, N, 2)
        e = adj_matrix.float().unsqueeze(-1)                               # (B, N, N, 1)
        
        # 归一化时间步
        norm_time = float(time_step) / float(GameConfig.EPISODE_LENGTH)
        
        # 优化：直接在 Device 上创建 Tensor，避免 CPU->GPU 拷贝
        # u shape: (B, 1)
        u = torch.full((B, 1), norm_time, device=device, dtype=torch.float32)
        
        # Mask (去对角线): 1.0 - Identity
        # 建议：如果 N 固定，mask 可以缓存以节省计算，但此处实时计算开销也可忽略
        mask = 1.0 - torch.eye(N, device=device).unsqueeze(0)
        
        # 2. Block 1 (Standard Message Passing)
        # 输入维度: V(2), E(1), U(1) -> 输出维度: All 128
        # G' = (u', V', E')
        v_prime, e_prime, u_prime = self.block1(v, e, u, mask)
        
        # 3. Block 2 (Modified Message Passing)
        # 输入维度: All 128
        # 输出: edge_logits (B, N, N, 2), state_value (B, 1)
        # 注意：这里 block2 内部实现了特殊的 Node Update (切断边特征输入)
        _, edge_logits, state_value = self.block2(v_prime, e_prime, u_prime, mask)
        
        return edge_logits, state_value
    

class SocialPlannerAgent_v2(nn.Module):
    def __init__(self):
        super().__init__()
        
        h_dim = ModelConfig.HIDDEN_DIM # Usually 128
        
        # --- Block 1 (Standard) ---
        # 注意这里的改动：edge_dim 强制设为 2，以接收 V2 环境的丰富特征
        self.block1 = StandardGraphNetBlock(
            node_dim=ModelConfig.NODE_IN_DIM,   # 2
            edge_dim=2,                         # 【修改】V2边特征维数变为 2: [Adj, Edge_Games]
            global_dim=ModelConfig.GLOBAL_IN_DIM, # 1
            hidden_dim=h_dim
        )
        
        # --- Block 2 (Modified) ---
        self.block2 = ModifiedGraphNetBlock(
            input_dim=h_dim,
            hidden_dim=h_dim,
            action_dim=2  # 输出 Policy Logits：[0: 建议低风险, 1: 建议高风险]
        )
        
        self.apply(init_weights)

    def forward(self, capital, prev_decisions, edge_features, time_step):
        """
        注意：V2 接收的是 edge_features 而不再是 adj_matrix
        edge_features 形状已经是 (B, N, N, 2)
        """
        B, N = capital.shape
        device = capital.device
        
        # 1. Feature Construction (G = u, V, E)
        v = torch.stack([capital.float(), prev_decisions.float()], dim=-1) # (B, N, 2)
        
        # 【修改】直接使用传入的 edge_features，无需再 unsqueeze
        e = edge_features.float() # (B, N, N, 2)
        
        # 归一化时间步
        norm_time = float(time_step) / float(GameConfig.EPISODE_LENGTH)
        u = torch.full((B, 1), norm_time, device=device, dtype=torch.float32)
        
        # Mask (去对角线)
        mask = 1.0 - torch.eye(N, device=device).unsqueeze(0)
        
        # 2. Message Passing
        v_prime, e_prime, u_prime = self.block1(v, e, u, mask)
        _, edge_logits, state_value = self.block2(v_prime, e_prime, u_prime, mask)
        
        return edge_logits, state_value




SocialPlannerAgent = SocialPlannerAgent_v1 if MODE == 0 else SocialPlannerAgent_v2