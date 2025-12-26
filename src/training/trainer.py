import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from src.environment.game_env import NetworkGameEnv
from src.agents.human_bots import HumanBot
from src.planner.gnn_model import SocialPlannerAgent
from src.planner.policy import SocialPlannerPolicy

class SocialPlannerTrainer:
    def __init__(self, 
                 num_players=16, 
                 lr=1e-3, 
                 gamma=0.99, 
                 entropy_coef=0.01):
        
        self.num_players = num_players
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # 1. 初始化三大件
        self.env = NetworkGameEnv(num_players=num_players)
        self.bots = HumanBot(num_players=num_players)
        
        # 2. 初始化 AI 模型
        # Node input: [Last Action (1), Current Wealth (1)] -> dim=2
        # Edge input: [Connected? (1)] -> dim=1
        # Global input: [Round Progress (1)] -> dim=1
        self.planner = SocialPlannerAgent(input_node_dim=2, 
                                          input_edge_dim=1, 
                                          input_global_dim=1,
                                          hidden_dim=64)
        
        self.policy = SocialPlannerPolicy()
        
        # 3. 优化器
        self.optimizer = optim.Adam(self.planner.parameters(), lr=lr)
        
        # 设备管理
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.planner.to(self.device)

    def feature_adapter(self, adj_matrix, last_actions, payoffs, current_round, max_rounds):
        """
        数据转换中心: 把环境的 Numpy 数据转成 GNN 的 PyTorch Tensor
        """
        batch_size = 1 # 目前我们只跑一个环境
        
        # A. 节点特征: [Batch, N, 2] -> (Action, Payoff)
        # 归一化 payoff 以帮助网络收敛
        norm_payoffs = payoffs / (np.max(np.abs(payoffs)) + 1e-5) 
        node_feats = np.stack([last_actions, norm_payoffs], axis=1) # [N, 2]
        x = torch.FloatTensor(node_feats).unsqueeze(0).to(self.device) # [1, N, 2]
        
        # B. 边特征: [Batch, N, N, 1] -> (Adjacency)
        edge_feats = adj_matrix.reshape(self.num_players, self.num_players, 1)
        edge_attr = torch.FloatTensor(edge_feats).unsqueeze(0).to(self.device)
        
        # C. 全局特征: [Batch, 1] -> (Progress)
        u = torch.FloatTensor([[current_round / max_rounds]]).to(self.device)
        
        return x, edge_attr, u

    def run_episode(self, max_rounds=15, train=True):
        """
        运行一整局游戏 (15 Rounds)
        """
        # 重置环境和 Bots
        self.env = NetworkGameEnv(self.num_players) # 重新生成随机图
        self.bots = HumanBot(self.num_players)      # 重置性格和记忆
        
        # 存储轨迹用于 RL 更新
        log_probs = []
        values = []
        rewards = []
        entropies = []
        
        # 初始状态
        current_payoffs = np.zeros(self.num_players)
        last_actions = np.zeros(self.num_players)
        
        total_cooperation_rate = 0
        
        for r in range(max_rounds):
            # --- Step 1: 准备数据 ---
            x, edge_attr, u = self.feature_adapter(
                self.env.adj_matrix, last_actions, current_payoffs, r, max_rounds
            )
            
            # --- Step 2: Planner 思考 (Forward Pass) ---
            edge_logits, value_est = self.planner(x, edge_attr, u)
            
            # --- Step 3: 采样动作 (Policy) ---
            # proposed_adj: Planner 建议的新连边矩阵 (0/1)
            proposed_adj_tensor, log_prob, entropy = self.policy.get_action(edge_logits, deterministic=not train)
            
            # 转回 numpy 给环境用
            proposed_adj = proposed_adj_tensor.squeeze(0).cpu().numpy()
            
            # --- Step 4: Bots 决定是否接受建议 ---
            # 这是一个细致的交互过程
            current_adj = self.env.adj_matrix
            final_adj = current_adj.copy()
            
            # 遍历所有可能的边，看哪里不一样
            # 注意: 这里简化了模拟，假设每回合 Planner 都可以建议任意修改
            # 实际上论文中 Bot 会逐个判断
            for i in range(self.num_players):
                for j in range(i + 1, self.num_players):
                    # 如果建议不一样 (比如建议断开或连接)
                    if proposed_adj[i][j] != current_adj[i][j]:
                        action_type = 1 if proposed_adj[i][j] == 1 else -1 # 1=Make, -1=Break
                        
                        # Bot i 决定
                        accept_i = self.bots.decide_acceptance(i, j, action_type, last_actions[j])
                        # Bot j 决定
                        accept_j = self.bots.decide_acceptance(j, i, action_type, last_actions[i])
                        
                        # 双方都同意才修改 (如果是加边需要双方同意，断边通常只需一方，这里简化为双方)
                        # 论文原文: "Players decide whether to accept or reject" -> 通常加边需双方，删边需单方
                        # 为了简化复现，我们假设双方都同意才生效 (Coordinated)
                        if accept_i and accept_j:
                            final_adj[i][j] = final_adj[j][i] = proposed_adj[i][j]
            
            # --- Step 5: 环境更新 & 游戏博弈 ---
            self.env.update_graph(final_adj)
            
            # Bots 玩游戏 (决定合作/背叛)
            actions = self.bots.decide_cooperation(self.env.adj_matrix, current_round=r)
            
            # 算分
            step_payoffs = self.env.calculate_payoffs(actions)
            
            # --- Step 6: 计算奖励 (Reward) ---
            # 目标: 最大化全组的总收益 (Group Welfare)
            group_reward = np.sum(step_payoffs)
            
            # 归一化奖励 (为了训练稳定)
            # 每个人每轮大致在 -1 到 1 之间，总分大致在 -16 到 16 之间
            normalized_reward = group_reward / self.num_players 
            
            # 存储
            log_probs.append(log_prob)
            values.append(value_est)
            rewards.append(normalized_reward)
            entropies.append(entropy)
            
            # 更新状态供下一轮使用
            current_payoffs = step_payoffs
            last_actions = actions
            total_cooperation_rate += np.mean(actions)

        # --- Training Update (如果是训练模式) ---
        loss_value = 0
        if train:
            loss_value = self.update_model(rewards, values, log_probs, entropies)
            
        return {
            "mean_cooperation": total_cooperation_rate / max_rounds,
            "total_reward": np.sum(rewards),
            "loss": loss_value
        }

    def update_model(self, rewards, values, log_probs, entropies):
        """
        执行 A2C 算法的梯度更新
        """
        # 1. 计算回报 (Returns) - 从后往前推
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        # 强制指定 dtype=torch.float (即 Float32)
        returns = torch.tensor(returns, dtype=torch.float).to(self.device)
        values = torch.cat(values).squeeze(-1) # [15]
        log_probs = torch.cat(log_probs)       # [15]
        entropies = torch.cat(entropies)       # [15]
        
        # 2. 计算优势 (Advantage)
        # Advantage = 实际回报 - 预测回报
        # detach() 是因为我们不想让梯度传回 Critic 导致不稳定
        advantage = returns - values.detach()
        
        # 3. 计算 Loss
        # Actor Loss: 鼓励 Advantage 高的动作
        actor_loss = -(log_probs * advantage).mean()
        
        # Critic Loss: 让预测更准 (MSE)
        critic_loss = F.mse_loss(values, returns)
        
        # Entropy Loss: 鼓励探索 (减去 entropy 表示最大化 entropy)
        entropy_loss = -entropies.mean()
        
        total_loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss
        
        # 4. 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(self.planner.parameters(), 0.5)
        self.optimizer.step()
        
        return total_loss.item()