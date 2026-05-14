# plot_coop_comparison.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from env.game import PublicGoodsGame
from config import GameConfig, MODE
from planners.baselines import StaticPlanner, RandomPlanner, ReactivePlanner
from planners.graphnet import GraphNetPlanner 

def run_batch_simulation(strategy_name, model_path=None, n_games=1000, device="cuda"):
    """
    运行批量模拟并返回每轮的平均合作率
    """
    # 1. 初始化环境 (利用 batch_size 实现并行)
    env = PublicGoodsGame(batch_size=n_games, device=device)
    
    # 2. 选择策略 (包含 GraphNet)
    if strategy_name == "static": 
        planner = StaticPlanner()
    elif strategy_name == "random": 
        planner = RandomPlanner()
    elif strategy_name == "reactive": 
        planner = ReactivePlanner()
    elif strategy_name == "graphnet":
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"[错误] 未找到模型: {model_path}")
        planner = GraphNetPlanner(model_path=model_path, device=device)
    else: raise ValueError("Unknown strategy")

    # 3. 记录容器
    coop_rates =[]

    # 4. 运行博弈 (Round 1-15)
    with torch.no_grad():
        # Reset (执行 Round 1)
        capital, prev_decisions, edge_features = env.reset()
        coop_rates.append(prev_decisions.float().mean().item())

        # Round 2 - 15
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            # 获取 Logits
            # 注意: GraphNet 需要传入 edge_features
            logits = planner.get_logits(capital, prev_decisions, edge_features, t + 1)
            
            # 环境 Step
            next_state, _, _, _ = env.step(logits)
            capital, prev_decisions, edge_features = next_state
            
            # 计算平均合作率
            coop_rates.append(prev_decisions.float().mean().item())

    return coop_rates

def plot_results(n_games, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[系统] 正在使用 {device} 运行 {n_games} 局平均测试...")

    # 配置策略列表
    strategies =["static", "random", "reactive", "graphnet"]
    colors = {"static": "#95a5a6", "random": "#f1c40f", "reactive": "#e67e22", "graphnet": "#27ae60"}
    labels = {"static": "Static", "random": "Random", "reactive": "Reactive", "graphnet": "GraphNet Agent"}

    plt.figure(figsize=(10, 6))
    
    for str_name in strategies:
        print(f"正在运行 {str_name} ...")
        # 传入 model_path，只对 graphnet 有效
        data = run_batch_simulation(str_name, model_path=model_path, n_games=n_games, device=device)
        
        # 绘图
        rounds = np.arange(1, GameConfig.EPISODE_LENGTH + 1)
        plt.plot(rounds, data, label=labels[str_name], 
                 color=colors[str_name], marker='o', markersize=4, linewidth=2)

    # 美化
    plt.title(f"Cooperation Rate Evolution ({n_games} Games Avg)\nMode: {'V2 (Rules)' if MODE==1 else 'V1 (Topol)'}", fontsize=14)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Average Cooperation Rate", fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    
    plt.savefig("coop_rate_comparison.png", dpi=300)
    print(f"\n✅ 绘图完成！图片已保存为: coop_rate_comparison.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="运行的游戏局数")
    parser.add_argument("--model_path", type=str, 
                        default="checkpoints/replicate_0/final_model.pth",
                        help="GraphNet模型路径")
    args = parser.parse_args()
    
    plot_results(n_games=args.n, model_path=args.model_path)