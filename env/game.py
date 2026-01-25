# env/game.py
import torch
from config import GameConfig
from env.bots import SimulatedBots

class PublicGoodsGame:
    def __init__(self, batch_size, device):
        self.bs = batch_size
        self.device = device
        self.n = GameConfig.N_PLAYERS
        self.bots = SimulatedBots(batch_size, device)
        self.current_round = 0
        
    def reset(self):
        
        self.current_round = 0
        
        # 1. 初始图生成 (Erdos-Renyi)
        rand = torch.rand(self.bs, self.n, self.n, device=self.device)
        adj = (rand < GameConfig.ERDOS_RENYI_P).float()
        adj = torch.triu(adj, 1)
        adj = adj + adj.transpose(1, 2)
        self.adj = adj
        
        # 2. 初始资金
        self.capital = torch.ones(self.bs, self.n, device=self.device) * GameConfig.INITIAL_CAPITAL
        
        # 3. 初始化决策记录 (用于 Bot 计算 Round 1 的行为，虽然 Round 1 只看 Theta)
        # Round 1 的逻辑回归只需要 Theta，不需要邻居行为，但保持数据结构完整
        self.prev_decisions = torch.zeros(self.bs, self.n, device=self.device)

        # ==========================================
        # 核心修改：在 Agent 介入前，先运行 Round 1
        # Text: "Every turn, the social planner observes... players' most recent decisions"
        # ==========================================
        
        # Bot 决策 (Round 0 -> 对应游戏第1轮)
        # 注意：此处传入 self.capital，Bot 内部会处理破产保护
        coop_decisions = self.bots.decide_cooperation(
            self.current_round, 
            self.adj, 
            self.prev_decisions,
            self.capital
        )
        
        # 结算 Round 1 收益
        self._apply_payoffs(coop_decisions)
        
        # 更新状态供 Agent 观察
        self.prev_decisions = coop_decisions
        
        return self._get_state()

    def step(self, action_logits):
        """
        严格遵循顺序：
        1. Graph Update (基于 Agent 建议)
        2. Game Play (基于新图)
        """
        # 进入下一轮
        self.current_round += 1
        
        # ==========================================
        # Phase 1: Planner Recommends & Graph Updates
        # Text: "Players decide whether to accept... resulting in changes to graph"
        # ==========================================
        
        # 1.1 解析 Agent 动作 (Logits -> Actions)
        triu_mask = torch.triu(torch.ones(self.n, self.n, device=self.device), 1)
        probs_change = torch.softmax(action_logits, dim=-1)[..., 1]
        dist = torch.distributions.Bernoulli(probs_change)
        actions_change = dist.sample() * triu_mask.unsqueeze(0)
        
        # 1.2 生成建议类型 (Add/Delete)
        rec_type = torch.zeros_like(self.adj)
        rec_type[(self.adj == 0) & (actions_change == 1)] = 1  # Add
        rec_type[(self.adj == 1) & (actions_change == 1)] = -1 # Delete
        rec_type = rec_type + rec_type.transpose(1, 2)
        
        # 1.3 Bot 接受/拒绝建议
        accepted_mask = self.bots.decide_acceptance(rec_type, self.prev_decisions)
        
        # 1.4 应用图变更
        final_change_mask = (accepted_mask == 1) & (actions_change == 1)
        new_adj = self.adj.clone()
        new_adj[final_change_mask.bool()] = 1 - new_adj[final_change_mask.bool()]
        new_adj = torch.triu(new_adj, 1) + torch.triu(new_adj, 1).transpose(1, 2)
        self.adj = new_adj
        
        # ==========================================
        # Phase 2: Next Turn Begins & Game Play
        # Text: "Subsequently, another turn begins... players choose to cooperate"
        # ==========================================
        
        # 2.1 Bot 决策 (基于更新后的图)
        coop_decisions = self.bots.decide_cooperation(
            self.current_round, 
            self.adj, 
            self.prev_decisions,
            self.capital
        )
        
        # 2.2 结算本轮收益
        self._apply_payoffs(coop_decisions)
        self.prev_decisions = coop_decisions

        # ==========================================
        # Phase 3: Reward & Output
        # ==========================================
        
        # 计算 Agent 奖励 (基于本轮博弈后的群体福利 - 改变图的惩罚)
        group_welfare = self.capital.mean(dim=1)
        
        total_possible_edges = self.n * (self.n - 1) / 2
        num_changes = actions_change.sum(dim=(1,2))
        penalty = GameConfig.PENALTY_WEIGHT_P * (num_changes / total_possible_edges)
        
        reward = group_welfare - penalty
        
        return self._get_state(), reward, dist, actions_change

    def _apply_payoffs(self, coop_decisions):
        """计算并更新资金"""
        degree = self.adj.sum(dim=2)
        
        # Cost: c * degree (如果合作)
        costs = GameConfig.COST_C * degree * coop_decisions
        
        # Benefit: b * sum(neighbor_coops)
        coop_exp = coop_decisions.unsqueeze(1).expand(-1, self.n, -1)
        benefits = (self.adj * coop_exp).sum(dim=2) * GameConfig.BENEFIT_B
        
        payoffs = benefits - costs
        self.capital += payoffs

    def _get_state(self):
        return self.capital, self.prev_decisions, self.adj