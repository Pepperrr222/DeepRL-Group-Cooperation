# training/trainer.py
import os
import torch
import torch.optim as optim
import csv # 新增：用于保存结果
from model.agent import SocialPlannerAgent
from env.game import PublicGoodsGame
from training.a2c import compute_a2c_loss
from config import TrainConfig, GameConfig

try:
    from config import MODE
except ImportError:
    MODE = 0

# ==========================================
# V1: 原始版本 (修改连线)
# ==========================================
class Trainer_v1:
    def __init__(self, agent_id=0, seed=42):
        self.agent_id = agent_id
        self.device = torch.device(TrainConfig.DEVICE if torch.cuda.is_available() else "cpu")
        
        self.ckpt_dir = f"checkpoints/replicate_{agent_id}"
        os.makedirs(self.ckpt_dir, exist_ok=True)

        torch.manual_seed(seed)
        self.agent = SocialPlannerAgent().to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=TrainConfig.LR)
        self.env = PublicGoodsGame(TrainConfig.BATCH_SIZE, self.device)
        
        self.triu_mask = torch.triu(torch.ones(GameConfig.N_PLAYERS, GameConfig.N_PLAYERS, device=self.device), diagonal=1)
        self.num_valid_edges = self.triu_mask.sum()

        # --- 新增：初始化 CSV 日志 ---
        self.log_path = os.path.join(self.ckpt_dir, "training_log.csv")
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "coop_rate", "avg_capital", "loss", "p_loss"])

    def save_checkpoint(self, episode, is_final=False):
        filename = "final_model.pth" if is_final else f"ckpt_ep_{episode}.pth"
        path = os.path.join(self.ckpt_dir, filename)
        torch.save(self.agent.state_dict(), path)

    def train(self):
        rounds_per_episode = GameConfig.EPISODE_LENGTH - 1
        print(f"Replicate {self.agent_id} starting (V1) on {self.device}...")
        
        for episode in range(1, TrainConfig.MAX_EPISODES + 1):
            capital, prev_decisions, adj = self.env.reset()
            ep_log_probs, ep_values, ep_rewards, ep_entropies = [], [], [],[]
            
            for t in range(rounds_per_episode):
                edge_logits, value_est = self.agent(capital, prev_decisions, adj, t + 1)
                next_state, reward, dist, actions_change = self.env.step(edge_logits)
                (capital, prev_decisions, adj) = next_state
                
                raw_log_probs = dist.log_prob(actions_change)
                raw_entropy = dist.entropy()
                mask_expanded = self.triu_mask.unsqueeze(0)
                joint_log_prob = (raw_log_probs * mask_expanded).sum(dim=(1, 2))
                avg_entropy = (raw_entropy * mask_expanded).sum(dim=(1, 2)) / self.num_valid_edges
                
                ep_log_probs.append(joint_log_prob)
                ep_values.append(value_est)
                ep_rewards.append(reward)
                ep_entropies.append(avg_entropy)
            
            loss, p_loss, v_loss, e_loss = compute_a2c_loss(ep_log_probs, ep_values, ep_rewards, ep_entropies)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
            self.optimizer.step()
            
            if episode % TrainConfig.LOG_INTERVAL == 0:
                avg_coop = prev_decisions.float().mean().item()
                avg_cap = capital.mean().item() # 计算均资
                
                # 打印到屏幕
                print(f"[Rep {self.agent_id}] Ep {episode:4d} | Coop: {avg_coop:6.2%} | Cap: {avg_cap:5.2f} | Loss: {loss.item():.3f}")
                
                # 保存到 CSV
                with open(self.log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([episode, avg_coop, avg_cap, loss.item(), p_loss])
            
            if episode % 50000 == 0:
                self.save_checkpoint(episode)
        self.save_checkpoint(episode, is_final=True)


# ==========================================
# V2: 机制设计版本 (规则改风险)
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

        # --- 新增：初始化 CSV 日志 ---
        self.log_path = os.path.join(self.ckpt_dir, "training_log.csv")
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "coop_rate", "avg_capital", "risk_rate", "loss", "p_loss"])

    def save_checkpoint(self, episode, is_final=False):
        filename = "final_model.pth" if is_final else f"ckpt_ep_{episode}.pth"
        path = os.path.join(self.ckpt_dir, filename)
        torch.save(self.agent.state_dict(), path)

    def train(self):
        rounds_per_episode = GameConfig.EPISODE_LENGTH - 1
        print(f"Replicate {self.agent_id} starting (V2) on {self.device}...")
        
        for episode in range(1, TrainConfig.MAX_EPISODES + 1):
            capital, prev_decisions, edge_features = self.env.reset()
            current_adj = edge_features[..., 0] 
            valid_edge_mask = torch.triu(current_adj, diagonal=1)
            num_valid_edges = valid_edge_mask.sum(dim=(1, 2)) + 1e-8 
            
            ep_log_probs, ep_values, ep_rewards, ep_entropies = [], [], [],[]
            
            for t in range(rounds_per_episode):
                edge_logits, value_est = self.agent(capital, prev_decisions, edge_features, t + 1)
                next_state, reward, dist, actions_change = self.env.step(edge_logits)
                (capital, prev_decisions, edge_features) = next_state
                
                raw_log_probs = dist.log_prob(actions_change)
                raw_entropy = dist.entropy()
                joint_log_prob = (raw_log_probs * valid_edge_mask).sum(dim=(1, 2))
                avg_entropy = (raw_entropy * valid_edge_mask).sum(dim=(1, 2)) / num_valid_edges
                
                ep_log_probs.append(joint_log_prob)
                ep_values.append(value_est)
                ep_rewards.append(reward)
                ep_entropies.append(avg_entropy)
            
            loss, p_loss, v_loss, e_loss = compute_a2c_loss(ep_log_probs, ep_values, ep_rewards, ep_entropies)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
            self.optimizer.step()
            
            if episode % TrainConfig.LOG_INTERVAL == 0:
                # --- 指标提取 ---
                avg_coop = prev_decisions.float().mean().item()
                avg_cap = capital.mean().item()
                
                # 计算高风险比例：edge_features idx 1 是规则(0:low, 1:high)
                # 只统计真实存在的边 (valid_edge_mask)
                rules = edge_features[..., 1]
                total_active_edges = valid_edge_mask.sum().item()
                high_risk_count = (rules * valid_edge_mask).sum().item()
                avg_risk_rate = high_risk_count / (total_active_edges + 1e-8)
                
                # 打印到屏幕
                print(f"[Rep {self.agent_id}] Ep {episode:4d} | Coop: {avg_coop:6.2%} | Risk: {avg_risk_rate:6.2%} | Cap: {avg_cap:5.2f} | Loss: {loss.item():.3f}")
                
                # 保存到 CSV
                with open(self.log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([episode, avg_coop, avg_cap, avg_risk_rate, loss.item(), p_loss])
            
            if episode % 50000 == 0:
                self.save_checkpoint(episode)

        self.save_checkpoint(episode, is_final=True)

Trainer = Trainer_v1 if int(MODE) == 0 else Trainer_v2