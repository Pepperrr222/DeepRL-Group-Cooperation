# env/bots.py
import torch
from config import BotConfig, GameConfig

class SimulatedBots:
    def __init__(self, batch_size, device):
        self.bs = batch_size
        self.device = device
        self.n_players = GameConfig.N_PLAYERS
        
        # 初始化玩家性格 (Theta)
        self.theta = torch.normal(
            BotConfig.MU_THETA, 
            BotConfig.SIGMA_THETA, 
            size=(self.bs, self.n_players), 
            device=self.device
        )

    def decide_cooperation(self, round_num, adj_matrix, prev_decisions, current_capital):
        """
        根据当前网络状态、性格以及资金状况决定是否合作。
        Added: current_capital (B, N)
        """
        # 1. 计算基于性格和社交环境的合作意愿 (Logits)
        x_s = adj_matrix.sum(dim=2) # Degree (B, N)
        
        prev_decisions_exp = prev_decisions.unsqueeze(1).expand(-1, self.n_players, -1)
        x_n = (adj_matrix * prev_decisions_exp).sum(dim=2) # Cooperating neighbors
        
        x_r = torch.zeros_like(x_s)
        mask_degree = x_s > 0
        x_r[mask_degree] = x_n[mask_degree] / x_s[mask_degree] # Cooperation rate

        if round_num == 0:
            logits = BotConfig.BETA_PRIME_0 + BotConfig.BETA_PRIME_1 * self.theta
        else:
            logits = (BotConfig.BETA_0 + 
                      BotConfig.BETA_1 * x_s + 
                      BotConfig.BETA_2 * x_n + 
                      BotConfig.BETA_3 * x_r + 
                      self.theta)
        
        probs = torch.sigmoid(logits)
        initial_decisions = torch.bernoulli(probs)
        
        # 2. 强制背叛逻辑 (Forced Defection) - 论文 Eq(2)
        # 合作成本 = c * degree
        potential_cost = GameConfig.COST_C * x_s
        
        # 如果当前资金 < 潜在合作成本，强制设为不合作(0)

        cannot_afford_mask = current_capital < potential_cost
        
        # 将无法支付成本的玩家决策强制置为 0
        final_decisions = initial_decisions.clone()
        final_decisions[cannot_afford_mask] = 0.0
        
        return final_decisions

    def decide_acceptance(self, recommendations, prev_decisions):
        """
        决定是否接受 Agent 的连线/断线建议。
        """
        B, N, _ = recommendations.shape
        accept_probs = torch.zeros_like(recommendations, dtype=torch.float)
        
        partner_prev = prev_decisions.unsqueeze(1).expand(-1, N, -1)
        
        for rec_val in [-1, 1]:
            for partner_act in [0, 1]:
                prob = BotConfig.ACCEPT_PROBS[(rec_val, partner_act)]
                mask = (recommendations == rec_val) & (partner_prev == partner_act)
                accept_probs[mask] = prob
        
        return torch.bernoulli(accept_probs)