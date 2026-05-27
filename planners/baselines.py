# planners/baselines.py
import torch
import math
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

class StaticHighRiskPlannerV2(BasePlanner):

    def get_logits(self, capital, prev_decisions, edge_features, round_num):
        B, N, _, _ = edge_features.shape
        logits = torch.zeros(B, N, N, 2, device=edge_features.device)
        logits[..., 0] = -10.0
        logits[..., 1] = 10.0 # 始终选择 Index 1 (高风险)
        return logits



class RandomPlannerV2(BasePlanner):
    """
    V2 随机规划师（严格修正版）：
    以 30% 的概率建议“改变现状”（即翻转当前的风险规则）。
    - 如果当前是低风险 (0)，30% 概率建议高风险 (1)，70% 建议保持低风险 (0)
    - 如果当前是高风险 (1)，30% 概率建议低风险 (0)，70% 建议保持高风险 (1)
    """
    def get_logits(self, capital, prev_decisions, edge_features, round_num):
        # 获取当前真实的规则状态 (0=低风险, 1=高风险)
        # Shape: (B, N, N)
        current_games = edge_features[..., 1] 
        
        # log(0.3) 和 log(0.7)
        log_change = math.log(0.3)
        log_keep = math.log(0.7)
        
        # --- 使用 torch.where 完美替代布尔索引，彻底杜绝维度报错 ---
        
        # 对于 Index 0 (建议低风险) 的打分：
        # 如果当前是低风险(0)，则建议 0 代表"保持" (log_keep)
        # 如果当前是高风险(1)，则建议 0 代表"改变" (log_change)
        logits_0 = torch.where(current_games == 0, log_keep, log_change)
        
        # 对于 Index 1 (建议高风险) 的打分：
        # 如果当前是低风险(0)，则建议 1 代表"改变" (log_change)
        # 如果当前是高风险(1)，则建议 1 代表"保持" (log_keep)
        logits_1 = torch.where(current_games == 0, log_change, log_keep)
        
        # 将分离计算好的 logits 沿着最后一个维度拼接
        # Shape 变为: (B, N, N, 2)
        logits = torch.stack([logits_0, logits_1], dim=-1)
        
        return logits

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
    StaticHighRiskPlanner = StaticHighRiskPlannerV2 
    MaxConnectivityPlanner = MaxRiskPlannerV2
    ReactivePlanner = ReactivePlannerV2
else: # V1 模式
    StaticPlanner = StaticPlannerV1
    RandomPlanner = RandomPlannerV1
    MaxConnectivityPlanner = MaxConnectivityPlannerV1
    ReactivePlanner = ReactivePlannerV1