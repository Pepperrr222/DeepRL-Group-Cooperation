# vis2.py
import torch
import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 导入项目组件
from config import GameConfig, BotConfig, MODE
from env.game import PublicGoodsGame
from planners.baselines import StaticPlanner, RandomPlanner, ReactivePlanner

def get_node_colors_and_labels(capital, decisions, adj, edge_games):
    """
    计算每个节点的颜色和显示标签
    颜色逻辑: 蓝色(合作), 红色(背叛), 灰色(破产强制背叛)
    """
    N = capital.shape[0]
    colors = []
    
    # 计算每个点的潜在最大成本 (破产保护逻辑)
    worst_loss_low = abs(GameConfig.LOW_RISK_MATRIX[1][0])
    worst_loss_high = abs(GameConfig.HIGH_RISK_MATRIX[1][0])
    
    # 潜在成本矩阵 (N, N)
    potential_loss_mat = edge_games * worst_loss_high + (1.0 - edge_games) * worst_loss_low
    # 只看有连接的边
    node_potential_costs = (potential_loss_mat * adj).sum(dim=1)
    
    for i in range(N):
        cap = capital[i].item()
        dec = decisions[i].item()
        cost = node_potential_costs[i].item()
        
        if cap < cost:
            colors.append('#95a5a6') # 灰色: 资金不足 (Insolvent)
        elif dec == 1:
            colors.append('#3498db') # 蓝色: 合作 (Coop)
        else:
            colors.append('#e74c3c') # 红色: 背叛 (Defect)
            
    return colors

def draw_round(round_num, capital, decisions, edge_features, strategy_name, save_dir, pos):
    """绘制并保存单轮图像"""
    adj = edge_features[..., 0]
    edge_games = edge_features[..., 1]
    
    plt.figure(figsize=(12, 10))
    G = nx.from_numpy_array(adj.cpu().numpy())
    
    # 1. 获取节点颜色和标签
    node_colors = get_node_colors_and_labels(capital, decisions, adj, edge_games)
    
    # 2. 处理边: 分为高风险和低风险两组进行绘制
    edges = G.edges()
    high_risk_edges = []
    low_risk_edges = []
    for u, v in edges:
        if edge_games[u, v] == 1:
            high_risk_edges.append((u, v))
        else:
            low_risk_edges.append((u, v))

    # 3. 绘图
    # 绘点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, edgecolors='black')
    
    # 绘边
    nx.draw_networkx_edges(G, pos, edgelist=high_risk_edges, width=2.5, edge_color='#2c3e50', style='solid', label='High Risk')
    nx.draw_networkx_edges(G, pos, edgelist=low_risk_edges, width=1.0, edge_color='#bdc3c7', style='dashed', alpha=0.6, label='Low Risk')

    # 4. 标签: ID 和 资金
    # ID 在中心
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='white', font_weight='bold')
    
    # 资金在上方
    label_pos = {k: (v[0], v[1] + 0.07) for k, v in pos.items()}
    cap_labels = {i: f"${capital[i]:.1f}" for i in range(len(capital))}
    nx.draw_networkx_labels(G, label_pos, labels=cap_labels, font_size=8, font_color='black')

    # 5. 图表装饰
    coop_rate = decisions.mean().item()
    avg_cap = capital.mean().item()
    hr_rate = (edge_games * adj).sum() / (adj.sum() + 1e-8)
    
    plt.title(f"Round {round_num:02d} | Strategy: {strategy_name}\nCoop Rate: {coop_rate:.1%} | Avg Capital: ${avg_cap:.2f} | High-Risk: {hr_rate:.1%}", fontsize=14)
    
    # 图例
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='Cooperator'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Defector'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6', markersize=10, label='Insolvent (Forced D)'),
        Line2D([0], [0], color='#2c3e50', lw=2, label='High Risk Rule'),
        Line2D([0], [0], color='#bdc3c7', lw=1, ls='--', label='Low Risk Rule')
    ]
    plt.legend(handles=custom_lines, loc='upper right')
    plt.axis('off')
    
    # 6. 保存
    filename = os.path.join(save_dir, f"round_{round_num:02d}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="reactive", choices=["static", "random", "reactive"])
    args = parser.parse_args()

    if MODE == 0:
        print("[错误] 当前 config.py 为 V1 模式，请先切换为 MODE = 1 (V2 规则模式)")
        return

    device = torch.device("cpu")
    save_dir = f"visuals_{args.strategy}"
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # 1. 初始化环境
    env = PublicGoodsGame(batch_size=1, device=device)
    
    # 2. 选择策略
    if args.strategy == "static": planner = StaticPlanner()
    elif args.strategy == "random": planner = RandomPlanner()
    else: planner = ReactivePlanner()

    # 3. 运行 Round 1
    capital, prev_decisions, edge_features = env.reset()
    
    # 计算固定布局
    G_init = nx.from_numpy_array(edge_features[0, ..., 0].numpy())
    pos = nx.spring_layout(G_init, seed=42, k=1.0) # k控制节点间距

    print(f"正在生成 {args.strategy} 策略的可视化图像...")
    
    # 绘制第一轮
    draw_round(1, capital[0], prev_decisions[0], edge_features[0], args.strategy, save_dir, pos)

    # 4. 运行 Round 2-15
    for t in range(GameConfig.EPISODE_LENGTH - 1):
        logits = planner.get_logits(capital, prev_decisions, edge_features, t+1)
        next_state, reward, _, _ = env.step(logits)
        capital, prev_decisions, edge_features = next_state
        
        draw_round(t + 2, capital[0], prev_decisions[0], edge_features[0], args.strategy, save_dir, pos)

    print(f"✅ 完成！图片已保存至文件夹: {save_dir}/")

if __name__ == "__main__":
    main()