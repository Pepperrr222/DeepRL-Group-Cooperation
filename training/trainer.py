# training/trainer.py
import os
import torch
import torch.optim as optim
from model.agent import SocialPlannerAgent
from env.game import PublicGoodsGame
from training.a2c import compute_a2c_loss
from config import TrainConfig, GameConfig

class Trainer:
    def __init__(self, agent_id=0, seed=42):
        # 1. 设置设备和种子
        self.agent_id = agent_id
        self.device = torch.device(TrainConfig.DEVICE if torch.cuda.is_available() else "cpu")
        
        # 2. 每个副本拥有独立的保存目录
        self.ckpt_dir = f"checkpoints/replicate_{agent_id}"
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 3. 初始化 Agent
        # 这里 seed 的设置确保了每个进程的初始参数不同
        torch.manual_seed(seed)
        self.agent = SocialPlannerAgent().to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=TrainConfig.LR)
        
        # 4. 初始化环境
        self.env = PublicGoodsGame(TrainConfig.BATCH_SIZE, self.device)
        
        # 5. 预计算掩码
        self.triu_mask = torch.triu(
            torch.ones(GameConfig.N_PLAYERS, GameConfig.N_PLAYERS, device=self.device), 
            diagonal=1
        )
        self.num_valid_edges = self.triu_mask.sum()

    def save_checkpoint(self, episode, is_final=False):
        filename = "final_model.pth" if is_final else f"ckpt_ep_{episode}.pth"
        path = os.path.join(self.ckpt_dir, filename)
        torch.save(self.agent.state_dict(), path)

    def train(self):
        # 计算总局数：5e7 总轮次 / 14 次Agent决策每局 (15轮游戏包含1轮初始和14轮Agent干预)
        # 论文原文：5 x 10^7 simulated game rounds
        rounds_per_episode = GameConfig.EPISODE_LENGTH - 1
        max_episodes = int(5e7 // rounds_per_episode)
        
        print(f"Replicate {self.agent_id} starting training on {self.device} for {max_episodes} episodes...")
        
        for episode in range(1, max_episodes + 1):
            capital, prev_decisions, adj = self.env.reset()
            ep_log_probs, ep_values, ep_rewards, ep_entropies = [], [], [], []
            
            for t in range(rounds_per_episode):
                edge_logits, value_est = self.agent(capital, prev_decisions, adj, t + 1)
                next_state, reward, dist, actions_change = self.env.step(edge_logits)
                (capital, prev_decisions, adj) = next_state
                
                # 损失计算
                raw_log_probs = dist.log_prob(actions_change)
                raw_entropy = dist.entropy()
                mask_expanded = self.triu_mask.unsqueeze(0)
                
                joint_log_prob = (raw_log_probs * mask_expanded).sum(dim=(1, 2))
                avg_entropy = (raw_entropy * mask_expanded).sum(dim=(1, 2)) / self.num_valid_edges
                
                ep_log_probs.append(joint_log_prob)
                ep_values.append(value_est)
                ep_rewards.append(reward)
                ep_entropies.append(avg_entropy)
            
            # A2C 更新
            loss, p_loss, v_loss, e_loss = compute_a2c_loss(ep_log_probs, ep_values, ep_rewards, ep_entropies)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
            self.optimizer.step()
            
            # 日志：仅副本 0 频繁打印，其他副本减少打印以防控制台混乱
            if episode % TrainConfig.LOG_INTERVAL == 0:
                if self.agent_id == 0 or episode % 1000 == 0:
                    avg_coop = prev_decisions.float().mean().item()
                    print(f"[Rep {self.agent_id}] Ep {episode} | Coop: {avg_coop:.1%} | Loss: {loss.item():.3f}")
            
            # 定期保存
            if episode % 50000 == 0:
                self.save_checkpoint(episode)

        self.save_checkpoint(episode, is_final=True)