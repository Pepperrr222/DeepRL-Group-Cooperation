# env/game.py
import torch
from config import GameConfig, MODE
from env.bots import SimulatedBots

# ==========================================
# 版本 1: 原始版本 (改变网络拓扑连线)
# ==========================================
class PublicGoodsGame_v1:
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
        
        # 3. 初始化决策记录
        self.prev_decisions = torch.zeros(self.bs, self.n, device=self.device)

        # Bot 决策 (Round 0 -> 对应游戏第1轮)
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
        # 进入下一轮
        self.current_round += 1
        
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
        
        # 2.1 Bot 决策
        coop_decisions = self.bots.decide_cooperation(
            self.current_round, 
            self.adj, 
            self.prev_decisions,
            self.capital
        )
        
        # 2.2 结算本轮收益
        self._apply_payoffs(coop_decisions)
        self.prev_decisions = coop_decisions

        # 3. 计算 Agent 奖励
        group_welfare = self.capital.mean(dim=1)
        total_possible_edges = self.n * (self.n - 1) / 2
        num_changes = actions_change.sum(dim=(1,2))
        penalty = GameConfig.PENALTY_WEIGHT_P * (num_changes / total_possible_edges)
        
        reward = group_welfare - penalty
        
        return self._get_state(), reward, dist, actions_change

    def _apply_payoffs(self, coop_decisions):
        degree = self.adj.sum(dim=2)
        costs = GameConfig.COST_C * degree * coop_decisions
        coop_exp = coop_decisions.unsqueeze(1).expand(-1, self.n, -1)
        benefits = (self.adj * coop_exp).sum(dim=2) * GameConfig.BENEFIT_B
        
        payoffs = benefits - costs
        self.capital += payoffs

    def _get_state(self):
        return self.capital, self.prev_decisions, self.adj
    


# ==========================================
class PublicGoodsGame_v2:
    def __init__(self, batch_size, device):
        self.bs = batch_size
        self.device = device
        self.n = GameConfig.N_PLAYERS
        self.bots = SimulatedBots(batch_size, device)
        self.current_round = 0
        
    def reset(self):
        self.current_round = 0
        
        # 1. 初始图生成 (此后在 V2 中 adj 永远不变)
        rand = torch.rand(self.bs, self.n, self.n, device=self.device)
        adj = (rand < GameConfig.ERDOS_RENYI_P).float()
        self.adj = torch.triu(adj, 1) + torch.triu(adj, 1).transpose(1, 2)
        
        # 2. 初始博弈规则矩阵 (0: 低风险, 1: 高风险)
        # 初始化所有边均为低风险
        self.edge_games = torch.zeros_like(self.adj)
        
        self.capital = torch.ones(self.bs, self.n, device=self.device) * GameConfig.INITIAL_CAPITAL
        self.prev_decisions = torch.zeros(self.bs, self.n, device=self.device)

        # 3. Round 1 预热 (注意：额外传入了 edge_games)
        coop_decisions = self.bots.decide_cooperation(
            self.current_round, 
            self.adj, 
            self.prev_decisions,
            self.capital,
            self.edge_games  # V2 新增参数，用于破产计算
        )
        
        self._apply_payoffs(coop_decisions)
        self.prev_decisions = coop_decisions
        
        return self._get_state()

    def step(self, action_logits):
        self.current_round += 1
        
        # 获取真实存在的边，作为有效建议的 Mask
        valid_edges_mask = torch.triu(self.adj, 1) 
        
        # 1. Agent 建议 (Logits -> 0 或 1)
        # 0: 建议该边玩低风险; 1: 建议该边玩高风险
        probs_high_risk = torch.softmax(action_logits, dim=-1)[..., 1]
        dist = torch.distributions.Bernoulli(probs_high_risk)
        recommended_games = dist.sample() * valid_edges_mask
        
        # 2. 翻译建议语义 (适配原版人类的接受概率表)
        # 升级到高风险(1) -> 类似 Add (+1)
        # 降级到低风险(0) -> 类似 Delete (-1)
        rec_type = torch.zeros_like(self.edge_games)
        
        # 想要升级：当前是0，建议是1
        upgrade_mask = (self.edge_games == 0) & (recommended_games == 1) & (valid_edges_mask == 1)
        rec_type[upgrade_mask] = 1.0
        
        # 想要降级：当前是1，建议是0
        downgrade_mask = (self.edge_games == 1) & (recommended_games == 0) & (valid_edges_mask == 1)
        rec_type[downgrade_mask] = -1.0
        
        # 保持对称
        rec_type = rec_type + rec_type.transpose(1, 2)
        
        # 3. Bot 根据语义 (-1/0/1) 和 对方上轮表现决定是否接受
        accepted_mask = self.bots.decide_acceptance(rec_type, self.prev_decisions)
        
        # 4. 应用游戏规则更新
        # 只有提出改变建议，且Bot接受了，才真正改变 edge_games
        final_change_mask = (accepted_mask == 1) & (rec_type != 0)
        
        new_edge_games = self.edge_games.clone()
        new_edge_games[final_change_mask.bool()] = recommended_games[final_change_mask.bool()]
        # 保持无向图的规则对称
        self.edge_games = torch.triu(new_edge_games, 1) + torch.triu(new_edge_games, 1).transpose(1, 2)
        
        # 5. 玩家在新规则下进行博弈决策
        coop_decisions = self.bots.decide_cooperation(
            self.current_round, 
            self.adj, 
            self.prev_decisions,
            self.capital,
            self.edge_games
        )
        
        self._apply_payoffs(coop_decisions)
        self.prev_decisions = coop_decisions

        # 6. 奖励计算 (群体资金 - 修改规则带来的摩擦成本)
        group_welfare = self.capital.mean(dim=1)
        
        # 注意：V2中，由于拓扑固定，惩罚的基数变成"有效边"的数量，而不是理论最大边数
        num_changes = final_change_mask.sum(dim=(1, 2))
        num_valid_edges = valid_edges_mask.sum(dim=(1, 2)) * 2 # 乘以2是因为上面求和是对全图求的，或者直接对上三角求和
        
        # 为防止全零网络导致的除零错误，加上 epsilon (1e-8)
        penalty = GameConfig.PENALTY_WEIGHT_P * (num_changes / (num_valid_edges + 1e-8))
        
        reward = group_welfare - penalty
        
        # 返回 dist 和 recommended_games 供 trainer 计算 A2C
        return self._get_state(), reward, dist, recommended_games

    def _apply_payoffs(self, coop_decisions):
        """V2 的新型查表结算逻辑"""
        B, N = self.bs, self.n
        
        # 构造动作网格 (B, N, N)
        # my_acts[b, i, j] 代表在 batch b 中，玩家 i 的动作
        my_acts = coop_decisions.unsqueeze(2).expand(-1, -1, N).long()
        # opp_acts[b, i, j] 代表在 batch b 中，玩家 j 的动作
        opp_acts = coop_decisions.unsqueeze(1).expand(-1, N, -1).long()
        
        payoff_low = torch.zeros(B, N, N, device=self.device)
        payoff_high = torch.zeros(B, N, N, device=self.device)
        
        # 遍历 2x2 矩阵的可能性赋值 (纯向量化操作)
        for i in [0, 1]:
            for j in [0, 1]:
                mask = (my_acts == i) & (opp_acts == j)
                payoff_low[mask] = GameConfig.LOW_RISK_MATRIX[i][j]
                payoff_high[mask] = GameConfig.HIGH_RISK_MATRIX[i][j]
                
        # 结合当前的 edge_games 状态 (1 为高风险，0 为低风险) 组合出最终的收益矩阵
        actual_payoff_matrix = payoff_high * self.edge_games + payoff_low * (1.0 - self.edge_games)
        
        # 只有存在连线 (adj=1) 的地方才进行结算
        node_payoffs = (actual_payoff_matrix * self.adj).sum(dim=2)
        
        self.capital += node_payoffs

    def _get_state(self):
        """
        V2 的边状态维度变成了 2 维：
        dim 0: 拓扑结构 (0/1，恒定不变)
        dim 1: 规则状态 (0/1，动态变化)
        """
        edge_features = torch.stack([self.adj, self.edge_games], dim=-1)
        return self.capital, self.prev_decisions, edge_features
# 导出正确的环境类
PublicGoodsGame = PublicGoodsGame_v1 if MODE == 0 else PublicGoodsGame_v2