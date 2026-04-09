# training/trainer.py
import os
import torch
import torch.optim as optim
from model.agent import SocialPlannerAgent
from env.game import PublicGoodsGame
from training.a2c import compute_a2c_loss
from config import TrainConfig, GameConfig

try:
    from config import MODE
except ImportError:
    MODE = 0

# ==========================================
# V1: 原始版本 (修改连线，全连接图Mask)
# ==========================================
class Trainer_v1:
    def __init__(self, agent_id=0, seed=42):
        # 1. 设置设备和种子
        self.agent_id = agent_id
        self.device = torch.device(TrainConfig.DEVICE if torch.cuda.is_available() else "cpu")
        
        # 2. 每个副本拥有独立的保存目录
        self.ckpt_dir = f"checkpoints/replicate_{agent_id}"
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 3. 初始化 Agent
        torch.manual_seed(seed)
        self.agent = SocialPlannerAgent().to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=TrainConfig.LR)
        
        # 4. 初始化环境
        self.env = PublicGoodsGame(TrainConfig.BATCH_SIZE, self.device)
        
        # 5. 预计算掩码 (V1 是全局固定的全连通图上三角)
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
        rounds_per_episode = GameConfig.EPISODE_LENGTH - 1
        print(f"Replicate {self.agent_id} starting training (V1) on {self.device} for {TrainConfig.MAX_EPISODES} episodes...")
        
        for episode in range(1, TrainConfig.MAX_EPISODES + 1):
            capital, prev_decisions, adj = self.env.reset()
            ep_log_probs, ep_values, ep_rewards, ep_entropies = [], [], [],[]
            
            for t in range(rounds_per_episode):
                edge_logits, value_est = self.agent(capital, prev_decisions, adj, t + 1)
                next_state, reward, dist, actions_change = self.env.step(edge_logits)
                (capital, prev_decisions, adj) = next_state
                
                # 损失计算 (V1 使用固定的 mask)
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
            
            # 日志打印
            if episode % TrainConfig.LOG_INTERVAL == 0:
                if self.agent_id == 0 or episode % 1000 == 0:
                    avg_coop = prev_decisions.float().mean().item()
                    print(f"[Rep {self.agent_id}] Ep {episode:4d} | Coop: {avg_coop:6.2%} | Loss: {loss.item():.3f}")
            
            # 定期保存
            if episode % 50000 == 0:
                self.save_checkpoint(episode)

        self.save_checkpoint(episode, is_final=True)


# ==========================================
# V2: 机制设计版本 (动态掩码，仅对真实存在的边求梯度)
# ==========================================
class Trainer_v2:
    def __init__(self, agent_id=0, seed=42):
        self.agent_id = agent_id
        self.device = torch.device(TrainConfig.DEVICE if torch.cuda.is_available() else "cpu")
        
        self.ckpt_dir = f"checkpoints/replicate_{agent_id}"
        os.makedirs(self.ckpt_dir, exist_ok=True)

        torch.manual_seed(seed)
        self.agent = SocialPlannerAgent().to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=TrainConfig.LR)
        self.env = PublicGoodsGame(TrainConfig.BATCH_SIZE, self.device)

    def save_checkpoint(self, episode, is_final=False):
        filename = "final_model.pth" if is_final else f"ckpt_ep_{episode}.pth"
        path = os.path.join(self.ckpt_dir, filename)
        torch.save(self.agent.state_dict(), path)

    def train(self):
        rounds_per_episode = GameConfig.EPISODE_LENGTH - 1
        print(f"Replicate {self.agent_id} starting training (V2) on {self.device} for {TrainConfig.MAX_EPISODES} episodes...")
        
        for episode in range(1, TrainConfig.MAX_EPISODES + 1):
            # V2 中，env.reset() 返回的第三个状态是 edge_features (B, N, N, 2)
            capital, prev_decisions, edge_features = self.env.reset()
            
            # --- 动态获取当前 Batch 的真实拓扑掩码 ---
            current_adj = edge_features[..., 0] 
            valid_edge_mask = torch.triu(current_adj, diagonal=1) # (B, N, N)
            num_valid_edges = valid_edge_mask.sum(dim=(1, 2)) + 1e-8 # (B,)
            
            ep_log_probs, ep_values, ep_rewards, ep_entropies = [], [], [],[]
            
            for t in range(rounds_per_episode):
                # 获取策略
                edge_logits, value_est = self.agent(capital, prev_decisions, edge_features, t + 1)
                
                # 环境交互
                next_state, reward, dist, actions_change = self.env.step(edge_logits)
                (capital, prev_decisions, edge_features) = next_state
                
                # --- 动态掩码损失计算 ---
                raw_log_probs = dist.log_prob(actions_change)
                raw_entropy = dist.entropy()
                
                # 乘上 valid_edge_mask：只保留真实有连线的边上的概率梯度
                joint_log_prob = (raw_log_probs * valid_edge_mask).sum(dim=(1, 2)) # (B,)
                avg_entropy = (raw_entropy * valid_edge_mask).sum(dim=(1, 2)) / num_valid_edges # (B,)
                
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
            
            # 日志
            if episode % TrainConfig.LOG_INTERVAL == 0:
                if self.agent_id == 0 or episode % 1000 == 0:
                    avg_coop = prev_decisions.float().mean().item()
                    print(f"[Rep {self.agent_id}] Ep {episode:4d} | Coop: {avg_coop:6.2%} | Loss: {loss.item():.3f} (P: {p_loss:.2f})")
            
            # 定期保存
            if episode % 50000 == 0:
                self.save_checkpoint(episode)

        self.save_checkpoint(episode, is_final=True)

# 根据 MODE 导出对应的 Trainer
Trainer = Trainer_v1 if int(MODE) == 0 else Trainer_v2