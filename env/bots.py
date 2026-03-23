import torch
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
            # Round 1: 使用修正后的高截距
            # 注意：Round 1 只有 theta 影响，或者你可以保留 beta_prime_1
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
        initial_decisions = torch.bernoulli(probs)
        
        # 3. 强制背叛 (资金不足)
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

    def decide_cooperation(self, round_num, adj_matrix, prev_decisions, current_capital, edge_games):
        """
        V2 版本的合作决策：需要多传入一个 edge_games 矩阵
        """
        # 1. 计算原始特征
        x_s = adj_matrix.sum(dim=2) # Degree
        
        prev_decisions_exp = prev_decisions.unsqueeze(1).expand(-1, self.n_players, -1)
        x_n = (adj_matrix * prev_decisions_exp).sum(dim=2) # Num Coop Neighbors
        
        x_r = torch.zeros_like(x_s)
        mask_degree = x_s > 0
        x_r[mask_degree] = x_n[mask_degree] / x_s[mask_degree] # Rate

        # 2. 计算 Logits (与 V1 保持一致的社会心理模型)
        if round_num == 0:
            logits = BotConfig.BETA_PRIME_0 + BotConfig.BETA_PRIME_1 * self.theta
        else:
            x_s_std = (x_s - BotConfig.MEAN_NEIGHBORS) / BotConfig.STD_NEIGHBORS
            x_n_std = (x_n - BotConfig.MEAN_COOP_NEIGHBORS) / BotConfig.STD_COOP_NEIGHBORS
            x_r_std = (x_r - BotConfig.MEAN_FRAC_COOP) / BotConfig.STD_FRAC_COOP
            
            logits = (BotConfig.BETA_0 + 
                      BotConfig.BETA_1 * x_s_std + 
                      BotConfig.BETA_2 * x_n_std + 
                      BotConfig.BETA_3 * x_r_std + 
                      self.theta)
        
        probs = torch.sigmoid(logits)
        initial_decisions = torch.bernoulli(probs)
        
        # 3. 强制背叛 (动态破产保护 - V2 核心逻辑)
        # 取两种博弈中，我合作(1)对方背叛(0)情况下的绝对损失值
        worst_loss_low = abs(GameConfig.LOW_RISK_MATRIX[1][0])
        worst_loss_high = abs(GameConfig.HIGH_RISK_MATRIX[1][0])
        
        # 构建当前每条边的潜在最大损失矩阵
        potential_loss_matrix = edge_games * worst_loss_high + (1.0 - edge_games) * worst_loss_low
        
        # 将邻接矩阵(adj_matrix)乘上去，只对真正连着的邻居计算损失，求和
        potential_cost = (potential_loss_matrix * adj_matrix).sum(dim=2)
        
        # 判断资金是否足以覆盖最坏情况
        cannot_afford_mask = current_capital < potential_cost
        
        final_decisions = initial_decisions.clone()
        final_decisions[cannot_afford_mask] = 0.0
        
        return final_decisions

    def decide_acceptance(self, recommendations, prev_decisions):
        """
        接受建议概率：完美复用 V1。
        在 game.py (v2) 中，Agent的升降级建议已经被翻译成 1(Add/Upgrade) 和 -1(Delete/Downgrade)
        """
        B, N, _ = recommendations.shape
        accept_probs = torch.zeros_like(recommendations, dtype=torch.float)
        
        partner_prev = prev_decisions.unsqueeze(1).expand(-1, N, -1)
        
        for rec_val in[-1, 1]:
            for partner_act in [0, 1]:
                prob = BotConfig.ACCEPT_PROBS[(rec_val, partner_act)]
                mask = (recommendations == rec_val) & (partner_prev == partner_act)
                accept_probs[mask] = prob
        
        return torch.bernoulli(accept_probs)
    
SimulatedBots = SimulatedBots_v1 if MODE == 0 else SimulatedBots_v2