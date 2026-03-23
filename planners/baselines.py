import torch
from .base import BasePlanner, prob_to_logits
from config import MODE

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
    


# planners/baselines.py (增加 V2 部分)

class StaticPlannerV2(BasePlanner):
    """
    V2 静态规划师：永远建议所有边玩‘低风险’博弈。
    代表没有任何干预的保守型社会。
    """
    def get_logits(self, capital, prev_decisions, edge_features, round_num):
        # 永远倾向于 index 0 (低风险)
        B, N, _, _ = edge_features.shape
        logits = torch.zeros(B, N, N, 2, device=edge_features.device)
        logits[..., 0] = 10.0  # Keep Low Risk
        logits[..., 1] = -10.0
        return logits

class RandomPlannerV2(BasePlanner):
    """
    V2 随机规划师：以 30% 的概率建议‘高风险’，70% 建议‘低风险’。
    """
    def get_logits(self, capital, prev_decisions, edge_features, round_num):
        p_high_risk = 0.3
        logits = prob_to_logits(p_high_risk, edge_features.device)
        return logits.view(1, 1, 1, 2).expand(edge_features.shape[0], edge_features.shape[1], edge_features.shape[2], -1)

class MaxRiskPlannerV2(BasePlanner):
    """
    V2 最大风险规划师 (对应 V1 的最大连接)：
    永远建议所有存在的边都玩‘高风险’博弈，试图榨取最大收益。
    """
    def get_logits(self, capital, prev_decisions, edge_features, round_num):
        # 永远倾向于 index 1 (高风险)
        B, N, _, _ = edge_features.shape
        logits = torch.zeros(B, N, N, 2, device=edge_features.device)
        logits[..., 0] = -10.0
        logits[..., 1] = 10.0 # Propose High Risk
        return logits
    

StaticPlanner, RandomPlanner, MaxConnectivityPlanner = StaticPlannerV2, RandomPlannerV2, MaxRiskPlannerV2 if MODE == 1 else (StaticPlanner, RandomPlanner, MaxConnectivityPlanner)