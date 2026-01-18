# training/trainer.py
import torch
import torch.optim as optim
from model.agent import SocialPlannerAgent
from env.game import PublicGoodsGame
from training.a2c import compute_a2c_loss
from config import TrainConfig, GameConfig

class Trainer:
    def __init__(self):
        self.device = torch.device(TrainConfig.DEVICE if torch.cuda.is_available() else "cpu")
        
        # Text E2: "parameter vectors ... let w be the concatenation of the entire set"
        # PyTorch optimizer handles the concatenated parameters naturally.
        self.agent = SocialPlannerAgent().to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=TrainConfig.LR)
        
        self.env = PublicGoodsGame(TrainConfig.BATCH_SIZE, self.device)
        
    def train(self):
        print(f"Start Training on {self.device}...")
        
        for episode in range(1, TrainConfig.MAX_EPISODES + 1):
            # 1. 轨迹收集初始化
            # reset() 现在包含 Round 1 的游戏过程
            capital, prev_decisions, adj = self.env.reset()
            
            # 存储轨迹数据 (Trajectory T)
            ep_log_probs = []
            ep_values = []
            ep_rewards = []
            ep_entropies = []
            
            # 2. 回合循环 (Round Loop)
            # 论文逻辑：Agent 介入 15 轮游戏。
            # 由于 reset() 跑了第 0 轮 (Round 1)，step() 需要跑剩下的 14 次 Agent 介入
            # Range: 0 to 14 (Total 15 interactions implied, but reset covers init)
            # Actually, paper says "Episode length 15". 
            # If reset plays R1, we need to play R2..R15. So loop 14 times.
            
            for t in range(GameConfig.EPISODE_LENGTH - 1):
                # A. Agent Observe & Output
                # Agent 观察当前状态 (st)，输出建议 (pi) 和 价值 (omega)
                # edge_logits (B, N, N, 2), state_value (B, 1)
                edge_logits, value_est = self.agent(capital, prev_decisions, adj, t)
                
                # B. Environment Step
                # 执行建议 -> 更新图 -> 运行下一轮游戏 -> 获得奖励
                next_state, reward, dist, actions_change = self.env.step(edge_logits)
                (capital, prev_decisions, adj) = next_state
                
                # C. Extract Log Probs & Entropy for Gradient
                # 动作空间是全图所有边的组合。
                # Log P(Joint Action) = Sum(Log P(Edge_ij))
                # 只计算上三角有效动作的 log_prob
                # actions_change is the binary mask of changes (B, N, N)
                
                # dist.log_prob 返回 (B, N, N)，对应每个位置采取 Change/Keep 的概率
                # 注意：actions_change 只有 0 或 1，对应 Keep 或 Change
                # 我们需要 actions_change 对应的 log_prob
                action_log_probs = dist.log_prob(actions_change)
                
                # 由于这是无向图且无自环，只对上三角求和作为整个图动作的 LogProb
                # dist is independent bernoulli for each edge
                joint_log_prob = action_log_probs.sum(dim=(1, 2)) / 2.0 
                
                # 同样的，熵也取平均或求和。论文说是 "regularize with entropy loss"
                # 通常取平均熵以保持尺度不变
                avg_entropy = dist.entropy().mean(dim=(1, 2))
                
                # 存储元组 (st, at, U_SPt) 的相关梯度信息
                ep_log_probs.append(joint_log_prob)
                ep_values.append(value_est)
                ep_rewards.append(reward)
                ep_entropies.append(avg_entropy)
            
            # 3. A2C 更新 (Update Step)
            # "Based on these trajectories, A2C proceeds by updating..."
            loss, p_loss, v_loss, e_loss = compute_a2c_loss(
                ep_log_probs, ep_values, ep_rewards, ep_entropies
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪 (Standard stability practice in RL)
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
            
            self.optimizer.step()
            
            # 4. 日志
            if episode % TrainConfig.LOG_INTERVAL == 0:
                avg_reward = torch.stack(ep_rewards).sum(dim=0).mean().item()
                avg_coop = prev_decisions.float().mean().item()
                print(f"Ep {episode:4d} | Reward: {avg_reward:.2f} | Coop Rate: {avg_coop:.2%}")
                print(f"         | Loss: {loss.item():.3f} (P: {p_loss:.3f}, V: {v_loss:.3f}, E: {e_loss:.3f})")