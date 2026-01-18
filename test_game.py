import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from env.game import PublicGoodsGame
from config import GameConfig, TrainConfig

def test_environment():
    # 1. 初始化环境
    device = torch.device("cpu") # 测试用 CPU 即可
    batch_size = 1 # 只观察 1 局游戏
    env = PublicGoodsGame(batch_size, device)
    
    capital, prev_decisions, adj = env.reset()
    
    # 用于记录数据
    history_coop_rate = []
    history_avg_capital = []
    snapshots = [] # 记录 (Round, Adj, Decisions) 用于画网络图
    
    print(f"=== 开始测试环境 (Batch Size: {batch_size}) ===")
    
    # 2. 运行 15 轮游戏
    for t in range(GameConfig.EPISODE_LENGTH):
        # 记录当前状态
        current_coop = prev_decisions[0].float().mean().item()
        current_cap = capital[0].mean().item()
        history_coop_rate.append(current_coop)
        history_avg_capital.append(current_cap)
        
        # 保存第 0, 7, 14 轮的快照用于画图
        if t in [0, 7, 14]:
            snapshots.append((t, adj[0].clone(), prev_decisions[0].clone()))
        
        # 生成随机动作 Logits (模拟一个未经训练的随机 Agent)
        # Shape: (B, N, N, 2) -> 随机生成数值，经过 softmax 后就是随机概率
        random_logits = torch.randn(batch_size, GameConfig.N_PLAYERS, GameConfig.N_PLAYERS, 2, device=device)
        
        # 环境步进
        (capital, prev_decisions, adj), reward, _, _ = env.step(random_logits, t)
        
        print(f"Round {t+1:02d}: Coop Rate = {current_coop:.2f}, Avg Capital = {current_cap:.2f}, Reward = {reward[0].item():.4f}")

    print("=== 测试结束 ===")
    return history_coop_rate, history_avg_capital, snapshots

def plot_metrics(coop, cap):
    """绘制合作率和资产曲线"""
    rounds = range(len(coop))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 合作率
    ax1.plot(rounds, coop, marker='o', color='tab:blue')
    ax1.set_title('Cooperation Rate over Time')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Rate (0-1)')
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True)
    
    # 资产
    ax2.plot(rounds, cap, marker='s', color='tab:green')
    ax2.set_title('Average Capital over Time')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Capital')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_network_snapshots(snapshots):
    """使用 NetworkX 绘制网络结构"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 固定节点位置，方便对比变化
    # 我们用第一帧的布局作为基准
    base_adj = snapshots[0][1].numpy()
    base_G = nx.from_numpy_array(base_adj)
    pos = nx.spring_layout(base_G, seed=42) 
    
    for i, (round_num, adj_tensor, decision_tensor) in enumerate(snapshots):
        ax = axes[i]
        
        # 转换数据
        adj_mat = adj_tensor.numpy()
        decisions = decision_tensor.numpy()
        
        # 建图
        G = nx.from_numpy_array(adj_mat)
        
        # 颜色映射: 1 (合作) -> 蓝色, 0 (背叛) -> 红色
        node_colors = ['#4A90E2' if d == 1 else '#E74C3C' for d in decisions]
        
        # 绘图
        nx.draw(G, pos, ax=ax, 
                with_labels=True, 
                node_color=node_colors, 
                edge_color='gray', 
                node_size=500, 
                font_color='white',
                font_size=8)
        
        ax.set_title(f"Round {round_num}\n(Blue=Coop, Red=Defect)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 运行模拟
    coop_data, cap_data, snaps = test_environment()
    
    # 画图 1: 数据曲线
    plot_metrics(coop_data, cap_data)
    
    # 画图 2: 网络结构图
    plot_network_snapshots(snaps)