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
            G = nx.random_regular_graph(d=degree, n=self.n)
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
            self.edge_games
        )
        
        self._apply_payoffs(coop_decisions)
        self.prev_decisions = coop_decisions
        
        return self._get_state()
    delta = BotConfig.DELTA
    def step(self, action_logits, delta):
        self.current_round += 1
        
        # 1. 提取有效边掩码 (上三角真实存在的边)
        valid_edges_mask = torch.triu(self.adj, 1) 
        
        # 2. Agent 输出建议的风险等级 (0: 低风险, 1: 高风险)
        probs_high_risk = torch.softmax(action_logits, dim=-1)[..., 1]
        dist = torch.distributions.Bernoulli(probs_high_risk)
        recommended_games = dist.sample() * valid_edges_mask
        delta=delta
        # 3. 将 0/1 建议翻译为 -1(降级), 0(不变), 1(升级)，对应 a_SP
        rec_type = torch.zeros_like(self.edge_games)
        
        # 想要升级：当前是0，建议是1
        upgrade_mask = (self.edge_games == 0) & (recommended_games == 1) & (valid_edges_mask == 1)
        rec_type[upgrade_mask] = 1.0
        
        # 想要降级：当前是1，建议是0
        downgrade_mask = (self.edge_games == 1) & (recommended_games == 0) & (valid_edges_mask == 1)
        rec_type[downgrade_mask] = -1.0
        
        # 对称化，使得 a_SP(i,j) = a_SP(j,i)
        rec_type = rec_type + rec_type.transpose(1, 2)
        
        # ==========================================
        # 核心修改 1：严格的接受逻辑 (基于要求：双方都接受才更改)
        # ==========================================
        # accept_i[b, i, j] 代表玩家 i 是否接受了对边 (i,j) 的建议
        accept_i = self.bots.decide_acceptance(rec_type, self.prev_decisions)
        # accept_j[b, i, j] 代表玩家 j 是否接受了对边 (i,j) 的建议
        accept_j = accept_i.transpose(1, 2)
        
        # 只有边两边的玩家都接受更改，才会更改边上的博弈
        both_accept = (accept_i == 1) & (accept_j == 1)
        
        # 最终改变：Agent提了建议 (rec_type != 0) 且 双方都接受 (both_accept)
        final_change_mask = (rec_type != 0) & both_accept & (valid_edges_mask == 1)
        
        # 更新规则
        new_edge_games = self.edge_games.clone()
        new_edge_games[final_change_mask.bool()] = recommended_games[final_change_mask.bool()]
        self.edge_games = torch.triu(new_edge_games, 1) + torch.triu(new_edge_games, 1).transpose(1, 2)
        
        # 4. 玩家在新规则下进行博弈决策
        coop_decisions = self.bots.decide_cooperation(
            self.current_round, 
            self.adj, 
            self.prev_decisions,
            self.capital,
            self.edge_games
        )
        
        self._apply_payoffs(coop_decisions)
        self.prev_decisions = coop_decisions

        # ==========================================
        # 核心修改 2：严格的惩罚函数 (公式 3 & 4)
        # ==========================================
        group_welfare = self.capital.mean(dim=1)
        
        # 计算 f(a_SP, a^1, i, j) = 1 的数量：
        # 条件：a_SP != 0 AND a^1_{i,j} == 0 AND a^1_{j,i} == 0
        both_reject = (accept_i == 0) & (accept_j == 0)
        penalty_mask = (rec_type != 0) & both_reject & (valid_edges_mask == 1)
        
        # 论文中的 m 是网络中可能存在的边。因为拓扑固定，m 实际上就是真实连接的边数
        num_penalties = penalty_mask.sum(dim=(1, 2))
        num_valid_edges = valid_edges_mask.sum(dim=(1, 2)) + 1e-8
        
        # 惩罚 = P * (1/m * SUM(f))
        penalty = GameConfig.PENALTY_WEIGHT_P * (num_penalties / num_valid_edges)
        
        # 恢复原版：纯粹的[群体平均资金 - 建议被双双拒绝的惩罚]
        reward = group_welfare - penalty
        
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