import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    可配置的 MLP：支持单层或多层，以及 ReLU/Tanh 激活
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, activation='tanh'):
        super(MLP, self).__init__()
        layers = []

        if activation == 'tanh':
            act_layer = nn.Tanh()
        elif activation == 'relu':
            act_layer = nn.ReLU()
        else:
            act_layer = None

        if num_layers <= 1:
            # 单层直接从 input -> output
            layers.append(nn.Linear(input_size, output_size))
            if act_layer is not None:
                layers.append(act_layer)
        else:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(act_layer)
            layers.append(nn.LayerNorm(hidden_size))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(act_layer)
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.Linear(hidden_size, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GraphNetBlock(nn.Module):
    """
    GraphNetBlock 支持独立的输入/输出维度，并允许节点更新选择是否使用更新后的边属性。
    Parameters:
      in_node_dim, in_edge_dim, in_global_dim: 输入维度
      out_node_dim, out_edge_dim, out_global_dim: 输出维度
      hidden_dim: MLP 的隐藏宽度（用于中间层，当 num_layers>1 时）
      num_layers: MLP 层数（默认为1，单层）
      activation_edge/node/global: 激活函数，或 None 表示无激活
      node_uses_updated_edge: 如果为 False，则节点更新使用原始 edge_attr 而不是 new_edge_attr
    """
    def __init__(self,
                 in_node_dim, in_edge_dim, in_global_dim,
                 out_node_dim, out_edge_dim, out_global_dim,
                 hidden_dim, num_layers=1,
                 activation_edge='tanh', activation_node='tanh', activation_global='tanh',
                 node_uses_updated_edge=True):
        super(GraphNetBlock, self).__init__()

        self.in_node_dim = in_node_dim
        self.in_edge_dim = in_edge_dim
        self.in_global_dim = in_global_dim
        self.out_node_dim = out_node_dim
        self.out_edge_dim = out_edge_dim
        self.out_global_dim = out_global_dim
        self.node_uses_updated_edge = node_uses_updated_edge

        # Edge update: input uses in_node_dim *2 + in_edge_dim + in_global_dim -> out_edge_dim
        self.edge_mlp = MLP(input_size=in_node_dim * 2 + in_edge_dim + in_global_dim,
                            hidden_size=hidden_dim,
                            output_size=out_edge_dim,
                            num_layers=num_layers,
                            activation=activation_edge)

        # Node update: input node + (edge dim used for node) + global -> out_node_dim
        edge_dim_for_node = out_edge_dim if node_uses_updated_edge else in_edge_dim
        self.node_mlp = MLP(input_size=in_node_dim + edge_dim_for_node + in_global_dim,
                            hidden_size=hidden_dim,
                            output_size=out_node_dim,
                            num_layers=num_layers,
                            activation=activation_node)

        # Global update: aggregate new node (out_node_dim) and new edge (out_edge_dim) + global(in_global_dim)
        self.global_mlp = MLP(input_size=out_node_dim + out_edge_dim + in_global_dim,
                              hidden_size=hidden_dim,
                              output_size=out_global_dim,
                              num_layers=num_layers,
                              activation=activation_global)

    def forward(self, x, edge_attr, u, batch_size, num_nodes):
        # x: [B, N, in_node_dim]
        # edge_attr: [B, N, N, in_edge_dim]
        # u: [B, in_global_dim]

        # --- A. Edge Update ---
        x_i = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        x_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        u_edge = u.view(batch_size, 1, 1, -1).expand(-1, num_nodes, num_nodes, -1)

        edge_inputs = torch.cat([x_i, x_j, edge_attr, u_edge], dim=-1)
        new_edge_attr = self.edge_mlp(edge_inputs)  # [B, N, N, out_edge_dim]

        # --- B. Node Update ---
        if self.node_uses_updated_edge:
            incoming_edges_agg = torch.sum(new_edge_attr, dim=2)
        else:
            incoming_edges_agg = torch.sum(edge_attr, dim=2)

        u_node = u.view(batch_size, 1, -1).expand(-1, num_nodes, -1)
        node_inputs = torch.cat([x, incoming_edges_agg, u_node], dim=-1)
        new_x = self.node_mlp(node_inputs)  # [B, N, out_node_dim]

        # --- C. Global Update ---
        node_agg = torch.sum(new_x, dim=1)
        edge_agg = torch.sum(new_edge_attr, dim=(1, 2))
        global_inputs = torch.cat([node_agg, edge_agg, u], dim=-1)
        new_u = self.global_mlp(global_inputs)  # [B, out_global_dim]

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
                 hidden_dim=128,
                 num_blocks=2):       
        super(SocialPlannerAgent, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # Encoders -> 映射到第一层 hidden (表中第一层为 128)
        self.node_encoder = nn.Linear(input_node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(input_edge_dim, hidden_dim)
        self.global_encoder = nn.Linear(input_global_dim, hidden_dim)

        # Core Processor: 两个专用 GraphNetBlock，严格对应论文表格
        # Module 1: outputs (edge=128, node=128, global=128), 使用 tanh
        block1 = GraphNetBlock(in_node_dim=hidden_dim, in_edge_dim=hidden_dim, in_global_dim=hidden_dim,
                               out_node_dim=hidden_dim, out_edge_dim=hidden_dim, out_global_dim=hidden_dim,
                               hidden_dim=hidden_dim, num_layers=1,
                               activation_edge='tanh', activation_node='tanh', activation_global='tanh',
                               node_uses_updated_edge=True)

        # Module 2: outputs (edge=2, node=128, global=1)
        # 按论文，node update in module2 不应使用更新后的边属性；
        # edge 和 global 在 module2 不使用激活（activation=None）
        block2 = GraphNetBlock(in_node_dim=hidden_dim, in_edge_dim=hidden_dim, in_global_dim=hidden_dim,
                               out_node_dim=hidden_dim, out_edge_dim=2, out_global_dim=1,
                               hidden_dim=hidden_dim, num_layers=1,
                               activation_edge=None, activation_node='tanh', activation_global=None,
                               node_uses_updated_edge=False)

        self.blocks = nn.ModuleList([block1, block2])

        # Policy head: 将 edge 的 2 维映射为单个 logit
        self.policy_head = nn.Linear(2, 1)

    def forward(self, x, edge_attr, u):
        batch_size, num_nodes, _ = x.size()
        
        # 1. Encode
        x = torch.tanh(self.node_encoder(x))
        edge_attr = torch.tanh(self.edge_encoder(edge_attr))
        u = torch.tanh(self.global_encoder(u))
        
        # 2. Process (Module1 then Module2)
        # Module1
        out_x1, out_edge1, out_u1 = self.blocks[0](x, edge_attr, u, batch_size, num_nodes)
        x = x + out_x1
        edge_attr = edge_attr + out_edge1
        u = u + out_u1

        # Module2 (note: block2 configured to not use updated edges for node update)
        out_x2, out_edge2, out_u2 = self.blocks[1](x, edge_attr, u, batch_size, num_nodes)
        x = x + out_x2
        # For module2, output dims differ: replace edge_attr/global with block outputs (no residual)
        edge_attr = out_edge2  # edge_attr now has out_edge_dim=2
        u = out_u2  # u now has out_global_dim=1

        # 3. Decode: edge_attr currently [B,N,N,2], global u is [B,1]
        edge_logits = self.policy_head(edge_attr).squeeze(-1)
        state_value = u
        
        # Mask diagonal (Self-loops are impossible)
        mask = torch.eye(num_nodes, device=x.device).bool().expand(batch_size, -1, -1)
        edge_logits.masked_fill_(mask, -float('inf'))
        
        return edge_logits, state_value