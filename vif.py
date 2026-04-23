# play_visual_forced.py
import torch
import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 导入配置和组件
from config import GameConfig, BotConfig, MODE
from env.game import PublicGoodsGame_v2
from planners.baselines import StaticPlanner, RandomPlanner, ReactivePlanner

# ==========================================
# 1. 定义强制采纳环境 (Patch)
# ==========================================
class ForcedPublicGoodsGame(PublicGoodsGame_v2):
    """
    修改后的环境类，强制忽略 Bot 的拒绝，100% 执行 Agent 建议
    """
    def step(self, action_logits):
        self.current_round += 1
        valid_edges_mask = torch.triu(self.adj, 1) 
        
        # Agent 建议 (0:低, 1:高)
        probs_high_risk = torch.softmax(action_logits, dim=-1)[..., 1]
        dist = torch.distributions.Bernoulli(probs_high_risk)
        recommended_games = dist.sample() * valid_edges_mask

        # --- 强制采纳逻辑 ---
        # 只要 Agent 提议了 (不管 Bot 愿不愿意)，直接覆盖
        # 找出 Agent 想要改变的边 (建议的规则与当前规则不同)
        change_mask = (recommended_games != self.edge_games) & (valid_edges_mask == 1)
        
        new_edge_games = self.edge_games.clone()
        new_edge_games[change_mask.bool()] = recommended_games[change_mask.bool()]
        self.edge_games = torch.triu(new_edge_games, 1) + torch.triu(new_edge_games, 1).transpose(1, 2)
        
        # 玩家博弈
        coop_decisions = self.bots.decide_cooperation(
            self.current_round, self.adj, self.prev_decisions, self.capital, self.edge_games
        )
        self._apply_payoffs(coop_decisions)
        self.prev_decisions = coop_decisions

        # 奖励计算
        group_welfare = self.capital.mean(dim=1)
        reward = group_welfare 
        
        return self._get_state(), reward, dist, recommended_games

# ==========================================
# 2. 状态检测与绘图函数
# ==========================================
def get_node_status(capital, decisions, adj, edge_games):
    """检测节点颜色：蓝(合作), 红(背叛), 灰(破产)"""
    N = capital.shape[0]
    colors = []
    
    worst_loss_low = abs(GameConfig.LOW_RISK_MATRIX[1][0])
    worst_loss_high = abs(GameConfig.HIGH_RISK_MATRIX[1][0])
    
    # 计算破产线
    potential_loss_mat = edge_games * worst_loss_high + (1.0 - edge_games) * worst_loss_low
    node_potential_costs = (potential_loss_mat * adj).sum(dim=1)
    
    for i in range(N):
        if capital[i] < node_potential_costs[i]:
            colors.append('#95a5a6') # 灰色: 破产
        elif decisions[i] == 1:
            colors.append('#3498db') # 蓝色: 合作
        else:
            colors.append('#e74c3c') # 红色: 背叛
    return colors

def draw_round(round_num, capital, decisions, edge_features, strategy_name, save_dir, pos):
    adj = edge_features[..., 0]
    edge_games = edge_features[..., 1]
    
    plt.figure(figsize=(12, 10))
    G = nx.from_numpy_array(adj.cpu().numpy())
    
    node_colors = get_node_status(capital, decisions, adj, edge_games)
    
    # 边样式
    high_edges = [(u, v) for u, v in G.edges() if edge_games[u, v] == 1]
    low_edges = [(u, v) for u, v in G.edges() if edge_games[u, v] == 0]

    # 绘图
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edgelist=high_edges, width=3.0, edge_color='#2c3e50', style='solid')
    nx.draw_networkx_edges(G, pos, edgelist=low_edges, width=1.0, edge_color='#bdc3c7', style='dashed', alpha=0.5)
    
    # 标签
    nx.draw_networkx_labels(G, pos, font_size=11, font_color='white', font_weight='bold')
    cap_pos = {k: (v[0], v[1] + 0.08) for k, v in pos.items()}
    cap_labels = {i: f"${capital[i]:.2f}" for i in range(len(capital))}
    nx.draw_networkx_labels(G, cap_pos, labels=cap_labels, font_size=9, font_color='#2c3e50')

    # 统计数据
    coop_rate = decisions.mean().item()
    hr_rate = (edge_games * adj).sum() / (adj.sum() + 1e-8)
    plt.title(f"Round {round_num:02d} | Strategy: {strategy_name.upper()} (FORCED)\nCoop Rate: {coop_rate:.1%} | High-Risk Edge: {hr_rate:.1%}", fontsize=15)
    
    # 图例
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='Cooperator'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Defector'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6', markersize=10, label='Insolvent (Bankrupt)'),
        Line2D([0], [0], color='#2c3e50', lw=2, label='High Risk (B=0.8, C=0.6)'),
        Line2D([0], [0], color='#bdc3c7', lw=1, ls='--', label='Low Risk (B=0.2, C=0.1)')
    ]
    plt.legend(handles=custom_lines, loc='upper right')
    plt.axis('off')
    
    plt.savefig(os.path.join(save_dir, f"round_{round_num:02d}.png"), bbox_inches='tight')
    plt.close()

# ==========================================
# 3. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="reactive", choices=["static", "random", "reactive"])
    args = parser.parse_args()

    device = torch.device("cpu")
    save_dir = f"forced_visuals_{args.strategy}"
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # 初始化强制环境
    env = ForcedPublicGoodsGame(batch_size=1, device=device)
    
    # 选择规划师
    if args.strategy == "static": planner = StaticPlanner()
    elif args.strategy == "random": planner = RandomPlanner()
    else: planner = ReactivePlanner()

    # 运行
    capital, decisions, edge_features = env.reset()
    pos = nx.spring_layout(nx.from_numpy_array(edge_features[0,...,0].numpy()), seed=42, k=1.0)

    print(f"正在生成 {args.strategy} 强制采纳模式的可视化图片...")
    draw_round(1, capital[0], decisions[0], edge_features[0], args.strategy, save_dir, pos)

    for t in range(GameConfig.EPISODE_LENGTH - 1):
        logits = planner.get_logits(capital, decisions, edge_features, t+1)
        next_state, _, _, _ = env.step(logits)
        capital, decisions, edge_features = next_state
        draw_round(t + 2, capital[0], decisions[0], edge_features[0], args.strategy, save_dir, pos)

    print(f"✅ 完成！图片已保存至: {save_dir}/")

if __name__ == "__main__":
    main()