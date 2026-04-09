# planners/baselines.py
import torch
from .base import BasePlanner, prob_to_logits
from config import MODE

# ==========================================================
# V1 原始类定义 (拓扑干预模式)
# ==========================================================

class StaticPlannerV1(BasePlanner):
    def get_logits(self, capital, prev_decisions, adj, round_num):
        B, N, _ = adj.shape
        logits = torch.zeros(B, N, N, 2, device=adj.device)
        logits[..., 0] = 10.0 # Keep
        logits[..., 1] = -10.0
        return logits

class RandomPlannerV1(BasePlanner):
    def get_logits(self, capital, prev_decisions, adj, round_num):
        p = 0.3
        logits = prob_to_logits(p, adj.device)
        return logits.view(1, 1, 1, 2).expand(adj.shape[0], adj.shape[1], adj.shape[2], -1)

class MaxConnectivityPlannerV1(BasePlanner):
    def get_logits(self, capital, prev_decisions, adj, round_num):
        p_change = torch.where(
            adj == 0,
            torch.tensor(0.999, device=adj.device),
            torch.tensor(0.001, device=adj.device)
        )
        return prob_to_logits(p_change, adj.device)

class ReactivePlannerV1(BasePlanner):
    """V1模式下的反应型逻辑：合作则连线，背叛则断开"""
    def get_logits(self, capital, prev_decisions, adj, round_num):
        B, N = prev_decisions.shape
        d_i = prev_decisions.unsqueeze(2); d_j = prev_decisions.unsqueeze(1)
        both_coop = (d_i == 1) & (d_j == 1)
        logits = torch.zeros(B, N, N, 2, device=adj.device)
        change_condition = (adj == 0) & both_coop | (adj == 1) & (~both_coop)
        logits[..., 1] = torch.where(change_condition, 10.0, -10.0)
        logits[..., 0] = torch.where(change_condition, -10.0, 10.0)
        return logits

# ==========================================================
# V2 扩展类定义 (机制设计模式)
# ==========================================================

class StaticPlannerV2(BasePlanner):
    def get_logits(self, capital, prev_decisions, edge_features, round_num):
        B, N, _, _ = edge_features.shape
        logits = torch.zeros(B, N, N, 2, device=edge_features.device)
        logits[..., 0] = 10.0  # Keep Low Risk
        logits[..., 1] = -10.0
        return logits

class RandomPlannerV2(BasePlanner):
    def get_logits(self, capital, prev_decisions, edge_features, round_num):
        p_high_risk = 0.3
        logits = prob_to_logits(p_high_risk, edge_features.device)
        return logits.view(1, 1, 1, 2).expand(edge_features.shape[0], edge_features.shape[1], edge_features.shape[2], -1)

class MaxRiskPlannerV2(BasePlanner):
    """V2模式下的最大风险逻辑：所有边都玩高风险"""
    def get_logits(self, capital, prev_decisions, edge_features, round_num):
        B, N, _, _ = edge_features.shape
        logits = torch.zeros(B, N, N, 2, device=edge_features.device)
        logits[..., 0] = -10.0
        logits[..., 1] = 10.0 # Propose High Risk
        return logits

class ReactivePlannerV2(BasePlanner):
    """
    V2反应型规划师 (信任升级与背叛降级):
    - 只有双方都合作 (C,C) 时，才建议高风险 (Index 1) -> 奖励
    - 只要有任何一方背叛，就建议低风险 (Index 0) -> 保护/制裁
    """
    def get_logits(self, capital, prev_decisions, edge_features, round_num):
        B, N = prev_decisions.shape
        d_i = prev_decisions.unsqueeze(2); d_j = prev_decisions.unsqueeze(1)
        
        # 判断是否双方都合作
        both_coop = (d_i == 1) & (d_j == 1)
        
        logits = torch.zeros(B, N, N, 2, device=prev_decisions.device)
        
        # 【修改处】：
        # 如果 both_coop 为 True，低风险(0)打负分，高风险(1)打正分
        # 如果 both_coop 为 False，低风险(0)打正分，高风险(1)打负分
        logits[..., 0] = torch.where(both_coop, -10.0, 10.0) 
        logits[..., 1] = torch.where(both_coop, 10.0, -10.0) 
        
        return logits

# ==========================================================
# 统一导出逻辑
# ==========================================================

if MODE == 1: # V2 模式
    StaticPlanner = StaticPlannerV2
    RandomPlanner = RandomPlannerV2
    MaxConnectivityPlanner = MaxRiskPlannerV2
    ReactivePlanner = ReactivePlannerV2
else: # V1 模式
    StaticPlanner = StaticPlannerV1
    RandomPlanner = RandomPlannerV1
    MaxConnectivityPlanner = MaxConnectivityPlannerV1
    ReactivePlanner = ReactivePlannerV1