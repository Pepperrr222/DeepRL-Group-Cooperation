# env/game.py
import torch
import networkx as nx
from config import GameConfig, MODE, BotConfig
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
# 版本 2: 机制设计版本 (RRG图 + 严谨接受逻辑 + 原版Reward)
# ==========================================
class PublicGoodsGame_v2:
    def __init__(self, batch_size, device):
        self.bs = batch_size
        self.device = device
        self.n = GameConfig.N_PLAYERS
        self.bots = SimulatedBots(batch_size, device)
        self.current_round = 0
        
        # --- 性能优化：预生成 Random Regular Graphs (RRG) 池 ---
        self.pool_size = 100
        self.rrg_pool = torch.zeros(self.pool_size, self.n, self.n, device=self.device)
        # degree 根据 config 动态获取，默认为 4
        degree = int(getattr(GameConfig, 'TARGET_AVG_DEGREE', 4))
        for i in range(self.pool_size):
            G = nx.random_regular_graph(degree, self.n)
            self.rrg_pool[i] = torch.tensor(nx.to_numpy_array(G), dtype=torch.float, device=self.device)

    def reset(self):
        self.current_round = 0
        
        # 1. 初始图生成 (从 RRG 池中随机采样)
        idx = torch.randint(0, self.pool_size, (self.bs,), device=self.device)
        adj = self.rrg_pool[idx]
        self.adj = torch.triu(adj, 1) + torch.triu(adj, 1).transpose(1, 2)
        
        # 2. 初始博弈规则矩阵 (0: 低风险, 1: 高风险)
        self.edge_games = torch.zeros_like(self.adj)
        
        self.capital = torch.ones(self.bs, self.n, device=self.device) * GameConfig.INITIAL_CAPITAL
        self.prev_decisions = torch.zeros(self.bs, self.n, device=self.device)

        # 3. Round 1 预热
        coop_decisions = self.bots.decide_cooperation(
            self.current_round, 
            self.adj, 
            self.prev_decisions,
            self.capital,
            self.edge_games,
            # Round 1 的 delta 传默认值即可，因为 Round 1 不涉及模仿
            delta=getattr(BotConfig, 'DELTA', 10.0) 
        )
        
        self._apply_payoffs(coop_decisions)
        self.prev_decisions = coop_decisions
        
        return self._get_state()

    # --- 修复处：将 delta 参数正确添加到函数签名，并提供回退默认值 ---
    def step(self, action_logits, delta=None):
        """
        V2 强制执行版：
        1. 删除了 accept_i/accept_j 逻辑，Agent 建议即现实。
        2. 奖励函数为纯群体平均资金（去除了拒绝惩罚）。
        """
        # 0. 处理 delta 默认值
        if delta is None:
            # 尝试从 BotConfig 获取，如果获取不到则默认为 10.0
            delta = getattr(BotConfig, 'DELTA', 10.0)
            
        self.current_round += 1
        
        # 1. 提取有效边掩码 (上三角且真实存在的边)
        valid_edges_mask = torch.triu(self.adj, 1) 
        
        # 2. Agent 输出建议的风险等级 (0: 低风险, 1: 高风险)
        # 根据 action_logits 采样，结果为 0 或 1
        probs_high_risk = torch.softmax(action_logits, dim=-1)[..., 1]
        dist = torch.distributions.Bernoulli(probs_high_risk)
        recommended_games = dist.sample() * valid_edges_mask

        # ==========================================
        # 核心修改：强制采纳 (Forced Compliance)
        # ==========================================
        # 找出 Agent 想要改变规则的边 (即建议的规则与当前 self.edge_games 不同的位置)
        # 注意：这里不再需要调用 bots.decide_acceptance
        final_change_mask = (recommended_games != self.edge_games) & (valid_edges_mask == 1)
        
        # 直接覆盖旧规则
        new_edge_games = self.edge_games.clone()
        new_edge_games[final_change_mask.bool()] = recommended_games[final_change_mask.bool()]
        
        # 保持无向图的规则对称
        self.edge_games = torch.triu(new_edge_games, 1) + torch.triu(new_edge_games, 1).transpose(1, 2)
        
        # 3. 玩家在【新规则】下进行博弈决策 (采用模仿动态逻辑)
        coop_decisions = self.bots.decide_cooperation(
            self.current_round, 
            self.adj, 
            self.prev_decisions,
            self.capital,
            self.edge_games,
            delta=delta # 传递 delta 参数
        )
        
        # 结算资金变动
        self._apply_payoffs(coop_decisions)
        self.prev_decisions = coop_decisions

        # ==========================================
        # 奖励计算：纯群体平均资金
        # ==========================================
        group_welfare = self.capital.mean(dim=1)
        

        reward = group_welfare 
        
        return self._get_state(), reward, dist, recommended_games

    def _apply_payoffs(self, coop_decisions):
        """V2 的新型查表结算逻辑"""
        B, N = self.bs, self.n
        
        my_acts = coop_decisions.unsqueeze(2).expand(-1, -1, N).long()
        opp_acts = coop_decisions.unsqueeze(1).expand(-1, N, -1).long()
        
        payoff_low = torch.zeros(B, N, N, device=self.device)
        payoff_high = torch.zeros(B, N, N, device=self.device)
        
        for i in[0, 1]:
            for j in [0, 1]:
                mask = (my_acts == i) & (opp_acts == j)
                payoff_low[mask] = GameConfig.LOW_RISK_MATRIX[i][j]
                payoff_high[mask] = GameConfig.HIGH_RISK_MATRIX[i][j]
                
        actual_payoff_matrix = payoff_high * self.edge_games + payoff_low * (1.0 - self.edge_games)
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