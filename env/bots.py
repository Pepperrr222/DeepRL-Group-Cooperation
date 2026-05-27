# env/bots.py
import torch
import torch.nn.functional as F
from config import BotConfig, GameConfig, MODE

class SimulatedBots_v1:
    def __init__(self, batch_size, device):
        self.bs = batch_size
        self.device = device
        self.n_players = GameConfig.N_PLAYERS
        
        self.theta = torch.normal(
            BotConfig.MU_THETA, 
            BotConfig.SIGMA_THETA, 
            size=(self.bs, self.n_players), 
            device=self.device
        )

    def decide_cooperation(self, round_num, adj_matrix, prev_decisions, current_capital):
        # 1. 计算原始特征
        x_s = adj_matrix.sum(dim=2) # Degree
        
        prev_decisions_exp = prev_decisions.unsqueeze(1).expand(-1, self.n_players, -1)
        x_n = (adj_matrix * prev_decisions_exp).sum(dim=2) # Num Coop Neighbors
        
        x_r = torch.zeros_like(x_s)
        mask_degree = x_s > 0
        x_r[mask_degree] = x_n[mask_degree] / x_s[mask_degree] # Rate

        # 2. 计算 Logits
        if round_num == 0:
            logits = BotConfig.BETA_PRIME_0 + BotConfig.BETA_PRIME_1 * self.theta
        else:
            # Round > 1: 标准化 + 修正后的系数
            x_s_std = (x_s - BotConfig.MEAN_NEIGHBORS) / BotConfig.STD_NEIGHBORS
            x_n_std = (x_n - BotConfig.MEAN_COOP_NEIGHBORS) / BotConfig.STD_COOP_NEIGHBORS
            x_r_std = (x_r - BotConfig.MEAN_FRAC_COOP) / BotConfig.STD_FRAC_COOP
            
            logits = (BotConfig.BETA_0 + 
                      BotConfig.BETA_1 * x_s_std + 
                      BotConfig.BETA_2 * x_n_std + 
                      BotConfig.BETA_3 * x_r_std + 
                      self.theta)
        
        probs = torch.sigmoid(logits)
        # 数值安全保护
        probs = torch.clamp(probs, 0.0, 1.0)
        initial_decisions = torch.bernoulli(probs)
        
        # 3. 强制背叛 (资金不足) - V1 保留此逻辑
        potential_cost = GameConfig.COST_C * x_s
        cannot_afford_mask = current_capital < potential_cost
        
        final_decisions = initial_decisions.clone()
        final_decisions[cannot_afford_mask] = 0.0
        
        return final_decisions

    def decide_acceptance(self, recommendations, prev_decisions):
        B, N, _ = recommendations.shape
        accept_probs = torch.zeros_like(recommendations, dtype=torch.float)
        
        partner_prev = prev_decisions.unsqueeze(1).expand(-1, N, -1)
        
        for rec_val in [-1, 1]:
            for partner_act in [0, 1]:
                prob = BotConfig.ACCEPT_PROBS[(rec_val, partner_act)]
                mask = (recommendations == rec_val) & (partner_prev == partner_act)
                accept_probs[mask] = prob
        
        # 数值安全保护
        accept_probs = torch.clamp(accept_probs, 0.0, 1.0)
        return torch.bernoulli(accept_probs)
    
class SimulatedBots_v2:
    def __init__(self, batch_size, device):
        self.bs = batch_size
        self.device = device
        self.n_players = GameConfig.N_PLAYERS
        
        self.theta = torch.normal(
            BotConfig.MU_THETA, 
            BotConfig.SIGMA_THETA, 
            size=(self.bs, self.n_players), 
            device=self.device
        )

    def decide_cooperation(self, round_num, adj_matrix, prev_decisions, current_capital, edge_games, delta=BotConfig.DELTA):
        """
        V2 演化博弈版本：模仿邻居策略
        已移除破产保护：即使资金为负也可继续合作，决策完全基于模仿逻辑。
        """
        B, N = self.bs, self.n_players
        
        if round_num == 0:
            logits = BotConfig.BETA_PRIME_0 + BotConfig.BETA_PRIME_1 * self.theta
            probs = torch.sigmoid(logits)
            probs = torch.clamp(probs, 0.0, 1.0) # 加上安全钳位
            initial_decisions = torch.bernoulli(probs)
        else:
            # 1. 计算上一轮的单步真实收益 u_j
            my_acts = prev_decisions.unsqueeze(2).expand(-1, -1, N).long()
            opp_acts = prev_decisions.unsqueeze(1).expand(-1, N, -1).long()
            
            payoff_low = torch.zeros(B, N, N, device=self.device)
            payoff_high = torch.zeros(B, N, N, device=self.device)
            
            for i in [0, 1]:
                for j in [0, 1]:
                    mask = (my_acts == i) & (opp_acts == j)
                    payoff_low[mask] = GameConfig.LOW_RISK_MATRIX[i][j]
                    payoff_high[mask] = GameConfig.HIGH_RISK_MATRIX[i][j]
                    
            actual_payoff_matrix = payoff_high * edge_games + payoff_low * (1.0 - edge_games)
            u = (actual_payoff_matrix * adj_matrix).sum(dim=2) 
            
            # 2. 演化博弈：计算模仿概率 p_ij
            A_prime = adj_matrix + torch.eye(N, device=self.device).unsqueeze(0)
            u_j = u.unsqueeze(1).expand(-1, N, -1)
            logits_imitation = u_j * delta
            logits_imitation = torch.where(A_prime == 1, logits_imitation, torch.tensor(-1e9, device=self.device))
            
            p_ij = F.softmax(logits_imitation, dim=2)
            
            # 3. 采样新策略
            prev_dec_j = prev_decisions.unsqueeze(1).expand(-1, N, -1) 
            prob_coop = (p_ij * prev_dec_j).sum(dim=2) # (B, N)
            
            # 数值安全保护 (防止浮点误差导致 p > 1)
            prob_coop = torch.clamp(prob_coop, 0.0, 1.0)
            
            initial_decisions = torch.bernoulli(prob_coop)

        # 【修改处】删除了原有的“强制背叛 (动态破产保护)”逻辑
        # 决策现在直接返回 initial_decisions，不检查 current_capital
        return initial_decisions

    def decide_acceptance(self, recommendations, prev_decisions):
        B, N, _ = recommendations.shape
        accept_probs = torch.zeros_like(recommendations, dtype=torch.float)
        
        partner_prev = prev_decisions.unsqueeze(1).expand(-1, N, -1)
        
        for rec_val in [-1, 1]:
            for partner_act in [0, 1]:
                prob = BotConfig.ACCEPT_PROBS[(rec_val, partner_act)]
                mask = (recommendations == rec_val) & (partner_prev == partner_act)
                accept_probs[mask] = prob
        
        # 数值安全保护
        accept_probs = torch.clamp(accept_probs, 0.0, 1.0)
        return torch.bernoulli(accept_probs)
    
SimulatedBots = SimulatedBots_v1 if MODE == 0 else SimulatedBots_v2