import numpy as np

class HumanBot:
    def __init__(self, num_players, mu_theta=-0.304, sigma_theta=2.410):
        self.num_players = num_players
        
        # 1. 初始化性格参数 theta_i (从正态分布采样)
        # 这里的 mu 和 sigma 论文中有提到是拟合真实数据的，我们先用默认值占位
        self.dispositions = np.random.normal(mu_theta, sigma_theta, num_players)
        
        # 2. 逻辑回归参数 (参考论文 Methods 部分的参数定义)
        # 这些参数决定了环境因素(邻居数量等)如何影响合作概率
        # 我们这里使用一组示例参数，实际复现时通常需要从真实数据拟合或使用论文附录的具体数值
        self.beta = {
            'beta0': -0.010,  # 截距
            'beta1': 0.193,   # 邻居总数 (xs) 的权重
            'beta2': 0.370,   # 合作邻居数 (xn) 的权重
            'beta3': 1.521    # 邻居合作率 (xr) 的权重
        }
        
        # 第一回合的参数 (因为第一回合没有历史数据)
        self.beta_initial = {
            'beta0': 1.807,
            'beta1': 0.818    # 作用于 theta
        }

        # 3. 接受建议的概率参数 (Phi)
        # 格式: [断开且对方背叛, 断开且对方合作, 连接且对方背叛, 连接且对方合作]
        # 论文中: 
        # phi0: break link given defect (通常概率较高)
        # phi1: break link given coop (通常概率较低)
        # phi2: make link given defect (通常概率较低)
        # phi3: make link given coop (通常概率较高)
        self.phi = [0.774, 0.085, 0.287, 0.909] 

        # 记录上一轮的动作，用于决策
        self.last_actions = np.zeros(num_players)

    def decide_cooperation(self, adj_matrix, current_round):
        """
        决定所有 Bot 在本回合是合作(1) 还是 背叛(0)
        """
        coop_probs = np.zeros(self.num_players)
        
        if current_round == 0:
            # 第一回合：主要基于性格
            logit = self.beta_initial['beta0'] + self.beta_initial['beta1'] * self.dispositions
        else:
            # 后续回合：基于环境统计量
            # xs: 邻居总数 (Degree)
            xs = np.sum(adj_matrix, axis=1)
            
            # xn: 合作的邻居数
            xn = np.dot(adj_matrix, self.last_actions)
            
            # xr: 邻居合作率 (处理除以0的情况)
            with np.errstate(divide='ignore', invalid='ignore'):
                xr = xn / xs
                xr = np.nan_to_num(xr, 0) # 如果没有邻居，合作率为0
            
            # 逻辑回归公式
            # logit = b0 + b1*xs + b2*xn + b3*xr + theta
            logit = (self.beta['beta0'] + 
                     self.beta['beta1'] * xs + 
                     self.beta['beta2'] * xn + 
                     self.beta['beta3'] * xr + 
                     self.dispositions)
            
        # Sigmoid 函数将 logit 转换为概率
        coop_probs = 1 / (1 + np.exp(-logit))
        
        # 根据概率采样动作 (0 或 1)
        actions = np.random.binomial(1, coop_probs)
        
        # 更新记录
        self.last_actions = actions
        return actions

    def decide_acceptance(self, u, v, action_type, partner_last_action):
        """
        决定是否接受 Planner 的建议
        u: 做出决定的玩家 ID
        v: 建议连接/断开的对象 ID
        action_type: -1 (断开) 或 1 (连接)
        partner_last_action: 对象 v 上一轮的动作 (0或1)
        """
        # 根据情况选择概率
        prob = 0.5 # Default
        
        if action_type == -1: # 建议断开 (Break)
            if partner_last_action == 0: # 对方背叛
                prob = self.phi[0]
            else: # 对方合作
                prob = self.phi[1]
        elif action_type == 1: # 建议连接 (Make)
            if partner_last_action == 0: # 对方背叛
                prob = self.phi[2]
            else: # 对方合作
                prob = self.phi[3]
                
        # 伯努利采样决定是否接受
        return np.random.random() < prob

# --- 简单测试 ---
if __name__ == "__main__":
    num = 16
    bots = HumanBot(num_players=num)
    
    # 模拟一个全连接的邻接矩阵
    adj = np.ones((num, num)) - np.eye(num)
    
    # 模拟第 0 回合
    print("Round 0 动作:", bots.decide_cooperation(adj, current_round=0))
    
    # 模拟第 1 回合
    print("Round 1 动作:", bots.decide_cooperation(adj, current_round=1))
    
    # 测试接受建议
    # 比如：建议玩家0 和 玩家1 连接，玩家1上一轮是合作的
    accepted = bots.decide_acceptance(u=0, v=1, action_type=1, partner_last_action=1)
    print(f"玩家0 是否接受与 合作者玩家1 连接的建议? {accepted}")