import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os

# 导入项目模块
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
N_GAMES = 15  # 模拟的局数 (对应图中半透明细线的数量)

# 颜色配置 (背景细线颜色, 前景均值颜色)
STYLES = {
    "static":           {"title": "Static network",          "c_line": "#FFC0CB", "c_mean": "#DC143C"}, # 粉/红
    "random":           {"title": "Random recommendations",  "c_line": "#F0E68C", "c_mean": "#B8860B"}, # 金/暗金
    "coop_clustering":  {"title": "Cooperative clustering",  "c_line": "#90EE90", "c_mean": "#228B22"}, # 浅绿/深绿
    "graphnet":         {"title": "GraphNet planner",        "c_line": "#87CEFA", "c_mean": "#1E90FF"}  # 天蓝/宝蓝
}

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

def get_final_capitals(strategy, n_games):
    """
    并行运行 n_games 局游戏，获取最后一轮每个人的资金
    Returns: numpy array shape (n_games, 16)
    """
    device = torch.device("cpu") # 绘图数据生成使用CPU即可
    planner = get_planner(strategy, device)
    
    if planner is None:
        return None

    # 初始化并行环境
    env = PublicGoodsGame(batch_size=n_games, device=device)
    capital, prev_decisions, adj = env.reset()
    
    # 跑到第 15 轮
    with torch.no_grad():
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            logits = planner.get_logits(capital, prev_decisions, adj, t + 1)
            next_state, _, _, _ = env.step(logits)
            capital, prev_decisions, adj = next_state
            
    return capital.numpy()

def compute_lorenz_curve(capitals):
    """
    计算单局游戏的洛伦兹曲线坐标
    Input: (16,) 资金数组
    Output: (x_coords, y_coords)
    """
    # 1. 排序 (从穷到富)
    sorted_caps = np.sort(capitals)
    
    # 2. 计算累积财富
    cum_caps = np.cumsum(sorted_caps)
    
    # 3. 归一化 (防止除以0)
    total_wealth = cum_caps[-1] + 1e-6
    y = cum_caps / total_wealth
    
    # 4. 在开头插入 (0,0) 点
    y = np.insert(y, 0, 0.0)
    
    # 5. 生成 X 轴 (0, 1/16, 2/16 ... 16/16)
    n = len(capitals)
    x = np.linspace(0.0, 1.0, n + 1)
    
    return x, y

def plot_fig2f():
    print("正在生成 Figure 2f (Lorenz Curves)...")
    strategies = ["static", "random", "coop_clustering", "graphnet"]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True, dpi=120)
    plt.subplots_adjust(wspace=0.05)
    
    # 遍历四个子图
    for i, strat in enumerate(strategies):
        ax = axes[i]
        style = STYLES[strat]
        print(f"  - 处理策略: {style['title']}")
        
        # 1. 获取数据
        # capitals_matrix: (N_GAMES, 16)
        capitals_matrix = get_final_capitals(strat, N_GAMES)
        
        if capitals_matrix is None:
            ax.text(0.5, 0.5, "Model Missing", ha='center')
            continue
            
        # 用于计算均值的累加器
        y_sum = np.zeros(GameConfig.N_PLAYERS + 1)
        
        # 2. 画单局曲线 (背景细线)
        for game_idx in range(N_GAMES):
            caps = capitals_matrix[game_idx]
            x, y = compute_lorenz_curve(caps)
            
            ax.plot(x, y, color=style['c_line'], alpha=0.4, linewidth=1.5)
            y_sum += y
            
        # 3. 画均值曲线 (前景粗线)
        y_mean = y_sum / N_GAMES
        # x 轴是固定的
        x_axis = np.linspace(0.0, 1.0, GameConfig.N_PLAYERS + 1)
        ax.plot(x_axis, y_mean, color=style['c_mean'], linestyle='-', linewidth=3)
        
        # 4. 画绝对平等线 (对角虚线)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 5. 样式调整
        ax.set_title(style['title'], loc='center', fontsize=12) # 论文标题居中
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        
        # 刻度设置
        ticks = [0, 0.25, 0.50, 0.75, 1]
        labels = ["0", "0.25", "0.50", "0.75", "1"]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        
        # 轴标签
        ax.set_xlabel("Share of group", fontsize=11)
        if i == 0:
            ax.set_ylabel("Share of capital", fontsize=11)
            # 在左上角添加 'f' 标签
            ax.text(-0.15, 1.1, 'f', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
            
        # 边框处理
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.tick_params(direction='out')

    # 保存
    save_path = "fig2_lorenz.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f" 图片已保存: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_fig2f()