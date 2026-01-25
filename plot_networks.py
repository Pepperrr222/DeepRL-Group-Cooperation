import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import sys
import os

# 导入必要模块
try:
    from env.game import PublicGoodsGame
    from planners import (
        StaticPlanner, RandomPlanner, CoopClusteringPlanner, GraphNetPlanner
    )
    from config import GameConfig
except ImportError as e:
    print(f"[错误] 无法导入模块: {e}")
    sys.exit(1)

# ================= 配置 =================
TARGET_ROUND = 10  # 论文通常展示第 10 轮或第 15 轮的状态
NODE_SCALE = 200   # 节点大小缩放系数

# 策略列表
STRATEGIES = [
    ("static",           "Static network"),
    ("random",           "Random recommendations"),
    ("coop_clustering",  "Cooperative clustering"),
    ("graphnet",         "GraphNet planner")
]

def get_planner(strategy, device):
    if strategy == "static": return StaticPlanner()
    if strategy == "random": return RandomPlanner()
    if strategy == "coop_clustering": return CoopClusteringPlanner()
    if strategy == "graphnet": 
        if os.path.exists("checkpoints/final_model.pth"):
            return GraphNetPlanner("checkpoints/final_model.pth", device)
        else:
            return None
    return None

def get_game_snapshot(strategy):
    """
    运行一局游戏，返回指定轮次的 (Adj, Decisions, Capital)
    """
    device = torch.device("cpu")
    planner = get_planner(strategy, device)
    
    if planner is None:
        return None, None, None

    # 初始化环境 (Batch=1)
    env = PublicGoodsGame(batch_size=1, device=device)
    capital, prev_decisions, adj = env.reset()
    
    # 运行到目标轮次
    # current_round 从 1 开始，循环运行直到 TARGET_ROUND
    # 我们需要在 TARGET_ROUND 结束后的状态
    
    with torch.no_grad():
        # 如果 TARGET_ROUND 是 1，直接返回 reset 后的状态
        if TARGET_ROUND > 1:
            for t in range(TARGET_ROUND - 1):
                # t+1 代表当前是第几轮的决策 (Round 1 已经过了)
                logits = planner.get_logits(capital, prev_decisions, adj, t + 1)
                next_state, _, _, _ = env.step(logits)
                capital, prev_decisions, adj = next_state
    
    return adj[0].numpy(), prev_decisions[0].numpy(), capital[0].numpy()

def plot_fig2e():
    print(f"正在生成 Figure 2e (Network Snapshots at Round {TARGET_ROUND})...")
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=120)
    plt.subplots_adjust(wspace=0.1)
    
    for i, (strat_key, strat_title) in enumerate(STRATEGIES):
        ax = axes[i]
        print(f"  - 模拟策略: {strat_title}")
        
        # 1. 获取数据
        adj, dec, cap = get_game_snapshot(strat_key)
        
        if adj is None:
            ax.text(0.5, 0.5, "Model Missing", ha='center')
            ax.axis('off')
            continue
            
        # 2. 构建图
        G = nx.from_numpy_array(adj)
        
        # 3. 设置样式
        # 颜色: 1=Coop(Blue), 0=Defect(Red)
        # 使用论文风格的柔和色调
        node_colors = ['#72A0C1' if d == 1 else '#F08080' for d in dec]
        
        # 大小: 基础大小 + 资金加成
        # GraphNet 下资金可能很多，做个限制防止遮挡
        node_sizes = [150 + c * NODE_SCALE for c in cap]
        
        # 布局算法
        # seed 固定布局的随机性，k 控制节点间距 (越大越松散)
        pos = nx.spring_layout(G, seed=42, k=0.7, iterations=50)
        
        # 4. 绘图
        # 画边
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6, width=1.5, edge_color='#444444')
        
        # 画点 (带黑色边框)
        nx.draw_networkx_nodes(G, pos, ax=ax, 
                               node_color=node_colors, 
                               node_size=node_sizes, 
                               edgecolors='black', 
                               linewidths=1.0)
        
        # 标题
        ax.set_title(strat_title, fontsize=12, pad=10)
        
        # 添加 'e' 标签到第一个图
        if i == 0:
            ax.text(-0.1, 1.1, 'e', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
            
        # 移除坐标轴
        ax.axis('off')

    # 保存
    save_path = "fig2_networks.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f" 图片已保存: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_fig2e()