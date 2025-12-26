import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    基础组件：多层感知机
    结构: Linear -> ReLU -> LayerNorm -> ...
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(hidden_size))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_size))
            
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GraphNetBlock(nn.Module):
    """
    GraphNet 核心模块：执行一次消息传递 (Message Passing)
    包含: Edge Update, Node Update, Global Update
    """
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super(GraphNetBlock, self).__init__()
        
        # 1. 边更新网络: 输入 (Sender + Receiver + Edge + Global)
        self.edge_mlp = MLP(input_size=node_dim * 2 + edge_dim + global_dim,
                            hidden_size=hidden_dim,
                            output_size=edge_dim)
        
        # 2. 节点更新网络: 输入 (Node + Aggregated_Edges + Global)
        self.node_mlp = MLP(input_size=node_dim + edge_dim + global_dim,
                            hidden_size=hidden_dim,
                            output_size=node_dim)
        
        # 3. 全局更新网络: 输入 (Aggregated_Nodes + Aggregated_Edges + Global)
        self.global_mlp = MLP(input_size=node_dim + edge_dim + global_dim,
                              hidden_size=hidden_dim,
                              output_size=global_dim)

    def forward(self, x, edge_attr, u, batch_size, num_nodes):
        # x: [batch, N, node_dim]
        # edge_attr: [batch, N, N, edge_dim]
        # u: [batch, global_dim]
        
        # --- A. Edge Update ---
        x_i = x.unsqueeze(2).expand(-1, -1, num_nodes, -1) # Sender
        x_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1) # Receiver
        u_edge = u.view(batch_size, 1, 1, -1).expand(-1, num_nodes, num_nodes, -1)
        
        edge_inputs = torch.cat([x_i, x_j, edge_attr, u_edge], dim=-1)
        new_edge_attr = self.edge_mlp(edge_inputs) 
        
        # --- B. Node Update ---
        # 聚合进来的边信息 (Sum over neighbors)
        incoming_edges_agg = torch.sum(new_edge_attr, dim=2) 
        u_node = u.view(batch_size, 1, -1).expand(-1, num_nodes, -1)
        
        node_inputs = torch.cat([x, incoming_edges_agg, u_node], dim=-1)
        new_x = self.node_mlp(node_inputs)
        
        # --- C. Global Update ---
        node_agg = torch.sum(new_x, dim=1)
        edge_agg = torch.sum(new_edge_attr, dim=(1, 2))
        
        global_inputs = torch.cat([node_agg, edge_agg, u], dim=-1)
        new_u = self.global_mlp(global_inputs)
        
        return new_x, new_edge_attr, new_u

class SocialPlannerAgent(nn.Module):
    """
    完整的 Social Planner 模型
    Input -> Encoder -> GraphNet Blocks -> Actor/Critic Heads -> Output
    """
    def __init__(self, 
                 input_node_dim=2,    
                 input_edge_dim=1,    
                 input_global_dim=1,  
                 hidden_dim=64,
                 num_blocks=2):       
        super(SocialPlannerAgent, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # Encoders
        self.node_encoder = nn.Linear(input_node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(input_edge_dim, hidden_dim)
        self.global_encoder = nn.Linear(input_global_dim, hidden_dim)
        
        # Core Processor
        self.blocks = nn.ModuleList([
            GraphNetBlock(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_blocks)
        ])
        
        # Decoders
        # Policy Head (Actor): 输出每条边的 Logits
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Value Head (Critic): 输出当前状态价值
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_attr, u):
        batch_size, num_nodes, _ = x.size()
        
        # 1. Encode
        x = F.relu(self.node_encoder(x))
        edge_attr = F.relu(self.edge_encoder(edge_attr))
        u = F.relu(self.global_encoder(u))
        
        # 2. Process (with Residual Connections)
        for block in self.blocks:
            out_x, out_edge, out_u = block(x, edge_attr, u, batch_size, num_nodes)
            x = x + out_x
            edge_attr = edge_attr + out_edge
            u = u + out_u
            
        # 3. Decode
        edge_logits = self.policy_head(edge_attr).squeeze(-1) # [B, N, N]
        state_value = self.value_head(u) # [B, 1]
        
        # Mask diagonal (Self-loops are impossible)
        mask = torch.eye(num_nodes, device=x.device).bool().expand(batch_size, -1, -1)
        edge_logits.masked_fill_(mask, -float('inf'))
        
        return edge_logits, state_value