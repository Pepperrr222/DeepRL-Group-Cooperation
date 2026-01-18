# model/agent.py
import torch
import torch.nn as nn
from model.graph_net import StandardGraphNetBlock, ModifiedGraphNetBlock
from config import ModelConfig, GameConfig
from utils.init_weights import init_weights

class SocialPlannerAgent(nn.Module):
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