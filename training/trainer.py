# training/trainer.py
import os
import torch
import torch.optim as optim
from model.agent import SocialPlannerAgent
from env.game import PublicGoodsGame
from training.a2c import compute_a2c_loss
from config import TrainConfig, GameConfig

class Trainer:
    def __init__(self):
        # 1. 设置计算设备
        self.device = torch.device(TrainConfig.DEVICE if torch.cuda.is_available() else "cpu")
        
        # 2. 初始化 Agent 和 优化器
        # 模型的参数会被自动移动到指定设备
        self.agent = SocialPlannerAgent().to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=TrainConfig.LR)
        
        # 3. 初始化游戏环境
        self.env = PublicGoodsGame(TrainConfig.BATCH_SIZE, self.device)
        
        # 4. 创建保存目录
        self.ckpt_dir = "checkpoints"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # 5. 预计算上三角掩码 (Upper Triangle Mask)
        # 用于在计算 Loss 时屏蔽掉对角线(自环)和下三角(重复边)
        # Shape: (N, N) -> 对角线及以下为0，上三角为1
        self.triu_mask = torch.triu(
            torch.ones(GameConfig.N_PLAYERS, GameConfig.N_PLAYERS, device=self.device), 
            diagonal=1
        )
        # 计算有效边的总数 (N*(N-1)/2)，用于 Entropy 的归一化
        self.num_valid_edges = self.triu_mask.sum()

    def save_checkpoint(self, episode, is_final=False):
        """保存模型参数到文件"""
        filename = "final_model.pth" if is_final else f"ckpt_ep_{episode}.pth"
        path = os.path.join(self.ckpt_dir, filename)
        torch.save(self.agent.state_dict(), path)
        # print(f"Model saved to {path}") # 可选：减少打印刷屏

    def train(self):
        print(f"Start Training on {self.device}...")
        
        for episode in range(1, TrainConfig.MAX_EPISODES + 1):
            # --- 1. 轨迹收集初始化 ---
            # reset() 内部已经运行了 Round 1 的博弈
            # 返回的是 Round 1 结束后的状态 (Capital, Prev_Decisions, Adj)
            capital, prev_decisions, adj = self.env.reset()
            
            # 用于存储本局游戏的轨迹数据
            ep_log_probs = []
            ep_values = []
            ep_rewards = []
            ep_entropies = []
            
            # --- 2. 回合循环 (Round 2 to 15) ---
            # reset 消耗了 Round 0 (即第1轮)。
            # 剩下的 14 轮 (GameConfig.EPISODE_LENGTH - 1) 由 Agent 介入。
            for t in range(GameConfig.EPISODE_LENGTH - 1):
                
                # A. Agent 决策
                # 传入 t+1 作为时间步 (代表当前是第2, 3...轮)
                edge_logits, value_est = self.agent(capital, prev_decisions, adj, t + 1)
                
                # B. 环境交互
                # dist: 伯努利分布对象，包含梯度信息
                # actions_change: 实际采样的动作 (0/1)
                # reward: Agent 的即时奖励
                next_state, reward, dist, actions_change = self.env.step(edge_logits)
                
                # 更新状态用于下一次循环
                (capital, prev_decisions, adj) = next_state
                
                # C. 提取有效的 Log Prob 和 Entropy (关键步骤)
                # 我们的图是无向图，且无自环。
                # 神经网络虽然输出了 N*N 个 Logits，但只有上三角部分是有效的动作。
                # 下三角只是重复信息或噪声，必须在计算梯度前屏蔽掉。
                
                # 1. 计算原始 Log Prob 和 Entropy (Shape: B, N, N)
                raw_log_probs = dist.log_prob(actions_change)
                raw_entropy = dist.entropy()
                
                # 2. 扩展掩码以匹配 Batch 维度 (Shape: 1, N, N)
                mask_expanded = self.triu_mask.unsqueeze(0)
                
                # 3. 计算联合动作的 Log Prob (Joint Log Probability)
                # P(Action) = P(edge_1) * P(edge_2) ... * P(edge_m)
                # Log P(Action) = Sum(Log P(edge_i))
                # 只对掩码为 1 的部分求和
                joint_log_prob = (raw_log_probs * mask_expanded).sum(dim=(1, 2)) # Shape: (B,)
                
                # 4. 计算平均 Entropy
                # 同样只计算有效边的 Entropy，并取平均值作为正则化项
                avg_entropy = (raw_entropy * mask_expanded).sum(dim=(1, 2)) / self.num_valid_edges # Shape: (B,)
                
                # 5. 存储数据
                ep_log_probs.append(joint_log_prob)
                ep_values.append(value_est)
                ep_rewards.append(reward)
                ep_entropies.append(avg_entropy)
            
            # --- 3. A2C 更新 (Update Step) ---
            # 一局游戏结束，利用收集的轨迹计算 Loss
            loss, p_loss, v_loss, e_loss = compute_a2c_loss(
                ep_log_probs, ep_values, ep_rewards, ep_entropies
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪：防止梯度爆炸，保持训练稳定
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
            
            self.optimizer.step()
            
            # --- 4. 日志与保存 ---
            if episode % TrainConfig.LOG_INTERVAL == 0:
                # 计算平均奖励和合作率用于展示
                avg_reward = torch.stack(ep_rewards).sum(dim=0).mean().item()
                avg_coop = prev_decisions.float().mean().item()
                
                print(f"Ep {episode:4d} | Reward: {avg_reward:6.2f} | Coop Rate: {avg_coop:6.2%} | Loss: {loss.item():6.3f} (P:{p_loss:5.2f} V:{v_loss:5.2f} E:{e_loss:5.2f})")
                
                # 定期保存 (例如每 500 轮)
                if episode % 100 == 0:
                    self.save_checkpoint(episode)

        # 训练结束保存最终模型
        self.save_checkpoint(episode, is_final=True)
        print("Training completed.")