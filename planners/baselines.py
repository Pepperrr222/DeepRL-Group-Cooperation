import torch
from .base import BasePlanner, prob_to_logits

class StaticPlanner(BasePlanner):
    def get_logits(self, capital, prev_decisions, adj, round_num):
        # 100% 保持现状
        B, N, _ = adj.shape
        logits = torch.zeros(B, N, N, 2, device=adj.device)
        logits[..., 0] = 10.0 # Keep
        logits[..., 1] = -10.0
        return logits

class RandomPlanner(BasePlanner):
    def get_logits(self, capital, prev_decisions, adj, round_num):
        # 30% 改变
        p = 0.3
        logits = prob_to_logits(p, adj.device)
        return logits.view(1, 1, 1, 2).expand(adj.shape[0], adj.shape[1], adj.shape[2], -1)

class MaxConnectivityPlanner(BasePlanner):
    def get_logits(self, capital, prev_decisions, adj, round_num):
        # 无连接则连，有连接则保
        p_change = torch.where(
            adj == 0, 
            torch.tensor(0.999, device=adj.device), 
            torch.tensor(0.001, device=adj.device)
        )
        return prob_to_logits(p_change, adj.device)