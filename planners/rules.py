import torch
import torch
from .base import BasePlanner, prob_to_logits

class CoopClusteringPlanner(BasePlanner):
    """
    Shirado & Christakis: 断开 C-D，连接 C-C
    """
    def get_logits(self, capital, prev_decisions, adj, round_num):
        B, N, _ = adj.shape
        device = adj.device
        
        d_i = prev_decisions.unsqueeze(2).expand(-1, -1, N)
        d_j = prev_decisions.unsqueeze(1).expand(-1, N, -1)
        
        is_cc = (d_i == 1) & (d_j == 1)
        is_cd = (d_i != d_j)
        
        # 基础噪音 5%
        p_change = torch.full((B, N, N), 0.05, device=device)
        
        # 规则
        target_add = is_cc & (adj == 0)
        p_change[target_add] = 0.9
        
        target_del = is_cd & (adj == 1)
        p_change[target_del] = 0.9
        
        return prob_to_logits(p_change, device)

class EncouragementPlanner(BasePlanner):
    """
    基于论文 Supp Table 7-9 的查表策略
    """
    def __init__(self):
        # Round 1-14
        self.cc_add = [1.0]*7 + [1.0, 1.0, 1.0, 1.0, 0.991, 0.954, 1.0]
        self.cc_del = [0.0]*7 + [0.01, 0.01, 0.011, 0.028, 0.035, 0.073, 0.108]
        
        self.cd_add = [0.993, 0.973, 0.914, 0.791, 0.644, 0.594, 0.463, 0.429, 0.366, 0.372, 0.361, 0.371, 0.328, 0.408]
        self.cd_del = [0.048, 0.029, 0.145, 0.213, 0.318, 0.508, 0.608, 0.745, 0.802, 0.753, 0.741, 0.774, 0.706, 0.722]
        
        self.dd_add = [0.0] * 14
        self.dd_del = [1.0] * 14

    def get_logits(self, capital, prev_decisions, adj, round_num):
        idx = min(round_num - 1, 13)
        B, N, _ = adj.shape
        device = adj.device
        
        p_add_mat = torch.zeros(B, N, N, device=device)
        p_del_mat = torch.zeros(B, N, N, device=device)
        
        d_i = prev_decisions.unsqueeze(2).expand(-1, -1, N)
        d_j = prev_decisions.unsqueeze(1).expand(-1, N, -1)
        
        is_cc = (d_i == 1) & (d_j == 1)
        is_dd = (d_i == 0) & (d_j == 0)
        is_cd = (d_i != d_j)
        
        # 填表
        p_add_mat[is_cc] = self.cc_add[idx]
        p_del_mat[is_cc] = self.cc_del[idx]
        p_add_mat[is_cd] = self.cd_add[idx]
        p_del_mat[is_cd] = self.cd_del[idx]
        p_add_mat[is_dd] = self.dd_add[idx]
        p_del_mat[is_dd] = self.dd_del[idx]
        
        p_change = torch.where(adj == 0, p_add_mat, p_del_mat)
        return prob_to_logits(p_change, device)

class NeutralPlanner(BasePlanner):
    """
    基于论文 Supp Table 10 的查表策略
    """
    def __init__(self):
        self.p_add = [0.891, 0.841, 0.656, 0.642, 0.608, 0.549, 0.545, 0.538, 0.520, 0.504, 0.532, 0.518, 0.529, 0.522]
        self.p_del = [0.119, 0.054, 0.084, 0.102, 0.117, 0.204, 0.215, 0.224, 0.239, 0.213, 0.215, 0.237, 0.232, 0.317]

    def get_logits(self, capital, prev_decisions, adj, round_num):
        idx = min(round_num - 1, 13)
        device = adj.device
        p_change = torch.where(
            adj == 0, 
            torch.tensor(self.p_add[idx], device=device), 
            torch.tensor(self.p_del[idx], device=device)
        )
        return prob_to_logits(p_change, device)