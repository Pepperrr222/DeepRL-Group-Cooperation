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
N_TRACES = 15        # 背景细线 (单局)
N_MEAN_GAMES = 15  # 均值计算 (大批量)
ROUNDS = np.arange(1, GameConfig.EPISODE_LENGTH + 1)

STRATEGIES = [
    ("static",           "Static network",      "#DC143C"),
    ("random",           "Random rec.",         "#DAA520"),
    ("coop_clustering",  "Coop. clustering",    "#76C758"),
    ("graphnet",         "GraphNet planner",    "#1E90FF"),
    ("encouragement",    "Encourag. planner",   "#E377C2"),
    ("neutral",          "Neutral planner",     "#9467BD"),
    ("max_connectivity", "Max. connectivity",   "#17BECF")
]

def calculate_gini_torch(capital_tensor):
    """
    向量化计算基尼系数 (Gini Coefficient)
    Input: (Batch, N_Players)
    Output: (Batch, )
    Formula: G = sum((2i - n - 1) * x_i) / (n * sum(x_i))  (x must be sorted)
    """
    # 1. 排序 (Sort capital values for each group)
    # values shape: (B, N)
    sorted_capital, _ = torch.sort(capital_tensor, dim=1)
    
    B, N = capital_tensor.shape
    device = capital_tensor.device
    
    # 2. 生成权重系数 index weights: (2i - n - 1)
    # i goes from 1 to N
    index = torch.arange(1, N + 1, device=device, dtype=torch.float32)
    weights = (2 * index - N - 1).unsqueeze(0) # (1, N)
    
    # 3. 计算分子分母
    numerator = (weights * sorted_capital).sum(dim=1)
    denominator = N * sorted_capital.sum(dim=1)
    
    # 防止除以0 (虽然本游戏中资金通常>0，但为了健壮性)
    gini = numerator / (denominator + 1e-8)
    
    return gini

def get_planner_instance(name, device):
    if name == "static": return StaticPlanner()
    if name == "random": return RandomPlanner()
    if name == "coop_clustering": return CoopClusteringPlanner()
    if name == "graphnet": 
        try: return GraphNetPlanner("checkpoints/final_model.pth", device)
        except: return None
    if name == "encouragement": return EncouragementPlanner()
    if name == "neutral": return NeutralPlanner()
    if name == "max_connectivity": return MaxConnectivityPlanner()
    return None

def get_inequality_data(strategy_name, batch_size):
    """
    运行游戏并返回每一轮的基尼系数
    Return Shape: (batch_size, 15)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planner = get_planner_instance(strategy_name, device)
    
    if planner is None: return None

    env = PublicGoodsGame(batch_size, device)
    gini_history = np.zeros((batch_size, GameConfig.EPISODE_LENGTH))
    
    # Round 1
    capital, prev_decisions, adj = env.reset()
    gini_history[:, 0] = calculate_gini_torch(capital).cpu().numpy()
    
    # Round 2-15
    with torch.no_grad():
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            logits = planner.get_logits(capital, prev_decisions, adj, t + 1)
            next_s, _, _, _ = env.step(logits)
            capital, prev_decisions, adj = next_s
            
            gini_history[:, t+1] = calculate_gini_torch(capital).cpu().numpy()
            
    return gini_history

def plot_inequality():
    print("开始绘制 Inequality (Gini Coefficient) ...")
    fig, axes = plt.subplots(1, 7, figsize=(22, 4), sharey=True, dpi=100)
    plt.subplots_adjust(wspace=0.05)
    
    for i, (strat_key, strat_title, color) in enumerate(STRATEGIES):
        ax = axes[i]
        print(f"  处理策略: {strat_key} ...")
        
        # 1. 获取数据
        traces = get_inequality_data(strat_key, N_TRACES)
        if traces is None:
            ax.text(0.5, 0.5, "Missing Model", ha='center')
            continue
            
        mean_batch = get_inequality_data(strat_key, N_MEAN_GAMES)
        mean_curve = mean_batch.mean(axis=0)
        
        # 2. 绘图
        # 背景细线
        for trace in traces:
            ax.plot(ROUNDS, trace, color=color, alpha=0.3, linewidth=1.2)
            
        # 均值虚线
        ax.plot(ROUNDS, mean_curve, color=color, linestyle='--', linewidth=2.5)
        
        # 3. 样式
        ax.set_title(strat_title, fontsize=11, fontweight='bold')
        ax.set_xlim(1, 15)
        ax.set_xticks([1, 5, 10, 15])
        # 根据截图，基尼系数大约在 0.05 到 0.35 之间
        ax.set_ylim(0.0, 0.4) 
        ax.grid(alpha=0.3)
        ax.set_xlabel("Round")
        
        if i == 0:
            ax.set_ylabel("Group inequality\nin capital (Gini)")

    plt.suptitle("d  Inequality of capital distribution within group", x=0.01, ha='left', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = "ext_fig1_inequality.png"
    plt.savefig(save_path)
    print(f" 图片已保存: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_inequality()