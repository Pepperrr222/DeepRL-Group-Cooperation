# plot_coop_comparison.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from env.game import PublicGoodsGame
from config import GameConfig, MODE, BotConfig
from planners.baselines import StaticPlanner, RandomPlanner, ReactivePlanner

def run_batch_simulation(strategy_name, n_games=10000, device="cuda"):
    """
    运行批量模拟并返回每轮的平均合作率
    """
    # 1. 初始化环境 (利用 batch_size 实现并行，速度极快)
    env = PublicGoodsGame(batch_size=n_games, device=device)
    
    # 2. 选择策略
    if strategy_name == "static": planner = StaticPlanner()
    elif strategy_name == "random": planner = RandomPlanner()
    elif strategy_name == "reactive": planner = ReactivePlanner()
    else: raise ValueError("Unknown strategy")

    # 3. 记录容器
    coop_rates = []

    # 4. 运行博弈
    with torch.no_grad():
        # Reset (执行 Round 1)
        capital, prev_decisions, edge_features = env.reset()
        coop_rates.append(prev_decisions.float().mean().item())

        # Round 2 - 15
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            logits = planner.get_logits(capital, prev_decisions, edge_features, t + 1)
            next_state, _, _, _ = env.step(logits)
            capital, prev_decisions, edge_features = next_state
            
            # 计算这一轮 10000 局游戏的平均合作率
            coop_rates.append(prev_decisions.float().mean().item())

    return coop_rates

def plot_results(n_games=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[系统] 正在使用 {device} 运行 {n_games} 局平均测试...")

    # 获取不同策略的数据
    strategies = ["static", "random", "reactive"]
    colors = {"static": "#95a5a6", "random": "#f1c40f", "reactive": "#e67e22"}
    labels = {"static": "Static (No Intervention)", 
              "random": "Random (30% Change)", 
              "reactive": "Reactive (C,C -> High Benefit)"}

    plt.figure(figsize=(10, 6))
    
    for str_name in strategies:
        print(f"正在运行 {str_name} ...")
        data = run_batch_simulation(str_name, n_games, device)
        
        # 绘图
        rounds = np.arange(1, GameConfig.EPISODE_LENGTH + 1)
        plt.plot(rounds, data, label=labels[str_name], 
                 color=colors[str_name], marker='o', markersize=4, linewidth=2)

    # 图表美化
    plt.title(f"Cooperation Rate Evolution ({n_games} Games Avg)\nMode: {'V2 (Mechanism Design)' if MODE==1 else 'V1'}", fontsize=14)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Average Cooperation Rate", fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(np.arange(1, 16))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # 保存并展示
    filename = f"coop_rate_mode_{MODE}.png"
    plt.savefig(filename, dpi=300)
    print(f"\n✅ 绘图完成！图片已保存为: {filename}")
    plt.show()

if __name__ == "__main__":
    # 如果 10000 局在你的显存中放不下，可以尝试 5000 或分批运行
    # 对于 N=20, 10000 局通常没问题；对于 N=100, 建议先试 1000
    try:
        plot_results(n_games=1000)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n[错误] 显存不足，请调小 n_games 数值。")
        else:
            raise e