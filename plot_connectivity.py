import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

# 导入必要模块
try:
    from env.game import PublicGoodsGame
    from planners import (
        StaticPlanner, RandomPlanner, CoopClusteringPlanner, GraphNetPlanner,
        EncouragementPlanner, NeutralPlanner, MaxConnectivityPlanner
    )
    from config import GameConfig
except ImportError as e:
    print(f"[错误] 无法导入模块: {e}")
    sys.exit(1)

# ================= 配置 =================
N_TRACES = 15        # 背景细线数量 (单局)
N_MEAN_GAMES = 10  # 均值计算局数 (大批量并行)
ROUNDS = np.arange(1, GameConfig.EPISODE_LENGTH + 1)
MAX_EDGES = GameConfig.N_PLAYERS * (GameConfig.N_PLAYERS - 1) / 2

# 策略列表与样式配置 (按照论文图表顺序)
STRATEGIES = [
    ("static",           "Static network",      "#DC143C"), # 红
    ("random",           "Random rec.",         "#DAA520"), # 金/黄
    ("coop_clustering",  "Coop. clustering",    "#76C758"), # 绿
    ("graphnet",         "GraphNet planner",    "#1E90FF"), # 蓝
    ("encouragement",    "Encourag. planner",   "#E377C2"), # 粉
    ("neutral",          "Neutral planner",     "#9467BD"), # 紫
    ("max_connectivity", "Max. connectivity",   "#17BECF")  # 青
]

def get_planner_instance(name, device):
    if name == "static": return StaticPlanner()
    if name == "random": return RandomPlanner()
    if name == "coop_clustering": return CoopClusteringPlanner()
    if name == "graphnet": 
        # 允许找不到模型时报错跳过
        try: return GraphNetPlanner("checkpoints/final_model.pth", device)
        except: return None
    if name == "encouragement": return EncouragementPlanner()
    if name == "neutral": return NeutralPlanner()
    if name == "max_connectivity": return MaxConnectivityPlanner()
    return None

def get_connectivity_data(strategy_name, batch_size):
    """
    运行游戏并返回每一轮的连接度 (Connectivity)
    Return Shape: (batch_size, 15)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planner = get_planner_instance(strategy_name, device)
    
    if planner is None:
        return None

    env = PublicGoodsGame(batch_size, device)
    # 存储结果
    connectivity_history = np.zeros((batch_size, GameConfig.EPISODE_LENGTH))
    
    # Round 1
    capital, prev_decisions, adj = env.reset()
    # 计算连接度: sum / 2 / 120
    connectivity_history[:, 0] = (adj.sum(dim=(1,2)) / 2.0 / MAX_EDGES).cpu().numpy()
    
    # Round 2-15
    with torch.no_grad():
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            logits = planner.get_logits(capital, prev_decisions, adj, t + 1)
            next_s, _, _, _ = env.step(logits)
            capital, prev_decisions, adj = next_s
            
            connectivity_history[:, t+1] = (adj.sum(dim=(1,2)) / 2.0 / MAX_EDGES).cpu().numpy()
            
    return connectivity_history

def plot_connectivity():
    print("开始绘制 Group Connectivity ...")
    fig, axes = plt.subplots(1, 7, figsize=(22, 4), sharey=True, dpi=100)
    plt.subplots_adjust(wspace=0.05)
    
    for i, (strat_key, strat_title, color) in enumerate(STRATEGIES):
        ax = axes[i]
        print(f"  处理策略: {strat_key} ...")
        
        # 1. 获取背景细线数据 (单局)
        traces = get_connectivity_data(strat_key, N_TRACES)
        
        if traces is None:
            ax.text(0.5, 0.5, "Missing Model", ha='center')
            continue
            
        # 2. 获取均值数据 (大批量)
        mean_batch = get_connectivity_data(strat_key, N_MEAN_GAMES)
        mean_curve = mean_batch.mean(axis=0)
        
        # 绘图 - 细线
        for trace in traces:
            ax.plot(ROUNDS, trace, color=color, alpha=0.3, linewidth=1.2)
            
        # 绘图 - 均值虚线
        ax.plot(ROUNDS, mean_curve, color=color, linestyle='--', linewidth=2.5)
        
        # 样式设置
        ax.set_title(strat_title, fontsize=11, fontweight='bold')
        ax.set_xlim(1, 15)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([1, 5, 10, 15])
        ax.grid(alpha=0.3)
        ax.set_xlabel("Round")
        
        if i == 0:
            ax.set_ylabel("Connectivity")
            
    plt.suptitle("b  Group connectivity", x=0.01, ha='left', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = "ext_fig1_connectivity.png"
    plt.savefig(save_path)
    print(f" 图片已保存: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_connectivity()