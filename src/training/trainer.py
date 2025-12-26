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
                 entropy_coef=0.01,
                 penalty_factor=0.5): # <--- [新增] 惩罚参数 P (默认设为 0.5)
        
        self.num_players = num_players
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.penalty_factor = penalty_factor # 存储 P
        
        # 1. 初始化三大件
        self.env = NetworkGameEnv(num_players=num_players)
        self.bots = HumanBot(num_players=num_players)
        
        # 2. 初始化 AI 模型
        self.planner = SocialPlannerAgent(input_node_dim=2, 
                                          input_edge_dim=1, 
                                          input_global_dim=1,
                                          hidden_dim=64)
        
        self.policy = SocialPlannerPolicy()
        
        # 设备管理
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == 'cuda':
            print(f"🚀 训练设备: GPU ({torch.cuda.get_device_name(0)})")
        else:
            print("⚠️ 训练设备: CPU (未检测到GPU，请检查环境)")

        self.planner.to(self.device)
        
        # 3. 优化器
        self.optimizer = optim.Adam(self.planner.parameters(), lr=lr)

    def feature_adapter(self, adj_matrix, last_actions, payoffs, current_round, max_rounds):
        # ... (数据转换部分保持不变) ...
        norm_payoffs = payoffs / (np.max(np.abs(payoffs)) + 1e-5) 
        node_feats = np.stack([last_actions, norm_payoffs], axis=1)
        x = torch.FloatTensor(node_feats).unsqueeze(0).to(self.device)
        edge_feats = adj_matrix.reshape(self.num_players, self.num_players, 1)
        edge_attr = torch.FloatTensor(edge_feats).unsqueeze(0).to(self.device)
        u = torch.FloatTensor([[current_round / max_rounds]]).to(self.device)
        return x, edge_attr, u

    def run_episode(self, max_rounds=15, train=True):
        self.env = NetworkGameEnv(self.num_players)
        self.bots = HumanBot(self.num_players)
        
        log_probs = []
        values = []
        rewards = []
        entropies = []
        
        current_payoffs = np.zeros(self.num_players)
        last_actions = np.zeros(self.num_players)
        
        total_cooperation_rate = 0
        
        for r in range(max_rounds):
            # --- Step 1-3: Planner 思考与采样 (略) ---
            x, edge_attr, u = self.feature_adapter(
                self.env.adj_matrix, last_actions, current_payoffs, r, max_rounds
            )
            
            edge_logits, value_est = self.planner(x, edge_attr, u)
            proposed_adj_tensor, log_prob, entropy = self.policy.get_action(edge_logits, deterministic=not train)
            proposed_adj = proposed_adj_tensor.squeeze(0).cpu().numpy()
            
            # --- Step 4: Bots 决定 & 拒绝率追踪 [关键修改点] ---
            current_adj = self.env.adj_matrix
            final_adj = current_adj.copy()
            
            # 追踪公式 (3) 和 (4) 中的变量
            num_suggestions = 0  # m: 总建议次数 (Planner Action != 0)
            num_rejections = 0   # sum(f): 被拒绝的次数 (Planner Action != 0 且 Bots Action == 0)

            for i in range(self.num_players):
                for j in range(i + 1, self.num_players):
                    # Planner 建议了改变 (a_SP != 0)
                    if proposed_adj[i][j] != current_adj[i][j]:
                        num_suggestions += 1 
                        
                        action_type = 1 if proposed_adj[i][j] == 1 else -1 # 1=Make, -1=Break
                        
                        accept_i = self.bots.decide_acceptance(i, j, action_type, last_actions[j])
                        accept_j = self.bots.decide_acceptance(j, i, action_type, last_actions[i])
                        
                        # 双方都同意才修改 (Bots Action != 0)
                        if accept_i and accept_j:
                            final_adj[i][j] = final_adj[j][i] = proposed_adj[i][j]
                        else:
                            # Planner 建议了改变，但 Bots 拒绝了 (Bots Action == 0)
                            # 满足惩罚条件 f = 1
                            num_rejections += 1

            # --- Step 5: 环境更新 & 游戏博弈 (略) ---
            self.env.update_graph(final_adj)
            actions = self.bots.decide_cooperation(self.env.adj_matrix, current_round=r)
            step_payoffs = self.env.calculate_payoffs(actions)
            
            # --- Step 6: 计算奖励 (公式 3) [关键修改点] ---
            
            # Term 1: 平均合作资本 (1/n * sum(d_i))
            avg_payoff = np.mean(step_payoffs)
            
            # Term 2: 惩罚项 P * (1/m * sum(f))
            if num_suggestions > 0:
                # 惩罚项 = P * 拒绝率
                rejection_rate = num_rejections / num_suggestions
                penalty_term = self.penalty_factor * rejection_rate
            else:
                # 没有建议，没有惩罚
                penalty_term = 0
            
            # 最终效用: U_sp = 平均收益 - 惩罚
            step_reward = avg_payoff - penalty_term
            
            # 存储轨迹
            log_probs.append(log_prob)
            values.append(value_est)
            rewards.append(step_reward) # 存储新的奖励
            entropies.append(entropy)
            
            current_payoffs = step_payoffs
            last_actions = actions
            total_cooperation_rate += np.mean(actions)

        # --- Training Update (略) ---
        loss_value = 0
        if train:
            loss_value = self.update_model(rewards, values, log_probs, entropies)
            
        return {
            "mean_cooperation": total_cooperation_rate / max_rounds,
            "total_reward": np.sum(rewards),
            "loss": loss_value
        }

    # ... (update_model 保持不变，它只负责 A2C 梯度计算) ...
    def update_model(self, rewards, values, log_probs, entropies):
        # 1. 计算回报 (Returns)
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float).to(self.device)
        values = torch.cat(values).squeeze(-1)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        
        # 2. 计算优势 (Advantage)
        advantage = returns - values.detach()
        
        # 3. 计算 Loss
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        
        total_loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss
        
        # 4. 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.planner.parameters(), 0.5)
        self.optimizer.step()
        
        return total_loss.item()