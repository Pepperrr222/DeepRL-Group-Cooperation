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
        # Inputs: Raw features
        # Outputs: Latent features (size h_dim)
        self.block1 = StandardGraphNetBlock(
            node_dim=ModelConfig.NODE_IN_DIM,   # 2
            edge_dim=ModelConfig.EDGE_IN_DIM,   # 1
            global_dim=ModelConfig.GLOBAL_IN_DIM, # 1
            hidden_dim=h_dim
        )
        
        # --- Block 2 (Modified) ---
        # Inputs: Latent features from Block 1 (all size h_dim)
        # Outputs: Policy Logits (size 2) and Value (size 1)
        self.block2 = ModifiedGraphNetBlock(
            input_dim=h_dim,
            hidden_dim=h_dim
        )
        
        self.apply(init_weights)

    def forward(self, capital, prev_decisions, adj_matrix, time_step):
        B, N = capital.shape
        device = capital.device
        
        # 1. Feature Construction (G = u, V, E)
        v = torch.stack([capital, prev_decisions], dim=-1) # (B, N, 2)
        e = adj_matrix.unsqueeze(-1)                       # (B, N, N, 1)
        
        norm_time = float(time_step) / float(GameConfig.EPISODE_LENGTH)
        u = torch.tensor([[norm_time]], device=device).expand(B, 1) # (B, 1)
        
        mask = 1.0 - torch.eye(N, device=device).unsqueeze(0)
        
        # 2. Block 1 (Standard Message Passing)
        # G' = (u', V', E')
        v_prime, e_prime, u_prime = self.block1(v, e, u, mask)
        
        # 3. Block 2 (Modified Message Passing)
        # Returns Policy Logits E'' and Value u'' directly
        edge_logits, state_value = self.block2(v_prime, e_prime, u_prime, mask)
        
        return edge_logits, state_value