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
N_TRACES = 15
N_MEAN_GAMES = 1000
ROUNDS = np.arange(1, GameConfig.EPISODE_LENGTH + 1)

# 同样的策略配置
STRATEGIES = [
    ("static",           "Static network",      "#DC143C"),
    ("random",           "Random rec.",         "#DAA520"),
    ("coop_clustering",  "Coop. clustering",    "#76C758"),
    ("graphnet",         "GraphNet planner",    "#1E90FF"),
    ("encouragement",    "Encourag. planner",   "#E377C2"),
    ("neutral",          "Neutral planner",     "#9467BD"),
    ("max_connectivity", "Max. connectivity",   "#17BECF")
]

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

def get_capital_data(strategy_name, batch_size):
    """
    运行游戏并返回每一轮的平均资金 (Mean Capital)
    Return Shape: (batch_size, 15)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planner = get_planner_instance(strategy_name, device)
    
    if planner is None: return None

    env = PublicGoodsGame(batch_size, device)
    capital_history = np.zeros((batch_size, GameConfig.EPISODE_LENGTH))
    
    capital, prev_decisions, adj = env.reset()
    capital_history[:, 0] = capital.mean(dim=1).cpu().numpy()
    
    with torch.no_grad():
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            logits = planner.get_logits(capital, prev_decisions, adj, t + 1)
            next_s, _, _, _ = env.step(logits)
            capital, prev_decisions, adj = next_s
            
            capital_history[:, t+1] = capital.mean(dim=1).cpu().numpy()
            
    return capital_history

def plot_capital():
    print("开始绘制 Average Accumulated Capital ...")
    fig, axes = plt.subplots(1, 7, figsize=(22, 4), sharey=True, dpi=100)
    plt.subplots_adjust(wspace=0.05)
    
    max_cap_seen = 0
    
    for i, (strat_key, strat_title, color) in enumerate(STRATEGIES):
        ax = axes[i]
        print(f"  处理策略: {strat_key} ...")
        
        traces = get_capital_data(strat_key, N_TRACES)
        if traces is None:
            ax.text(0.5, 0.5, "Missing Model", ha='center')
            continue
            
        mean_batch = get_capital_data(strat_key, N_MEAN_GAMES)
        mean_curve = mean_batch.mean(axis=0)
        
        # 记录最大值用于动态调整Y轴（可选）
        max_cap_seen = max(max_cap_seen, traces.max())
        
        for trace in traces:
            ax.plot(ROUNDS, trace, color=color, alpha=0.3, linewidth=1.2)
            
        ax.plot(ROUNDS, mean_curve, color=color, linestyle='--', linewidth=2.5)
        
        ax.set_title(strat_title, fontsize=11, fontweight='bold')
        ax.set_xlim(1, 15)
        ax.set_xticks([1, 5, 10, 15])
        ax.grid(alpha=0.3)
        ax.set_xlabel("Round")
        
        if i == 0:
            ax.set_ylabel("Mean capital level")

    # 稍微调整 Y 轴上限，留点空间
    axes[0].set_ylim(0.5, max_cap_seen * 1.1)
            
    plt.suptitle("c  Average accumulated capital in group", x=0.01, ha='left', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = "ext_fig1_capital.png"
    plt.savefig(save_path)
    print(f" 图片已保存: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_capital()