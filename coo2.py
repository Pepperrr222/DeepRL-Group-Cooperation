# plot_coop_comparison.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import gc
from env.game import PublicGoodsGame
from config import GameConfig, MODE
from planners.baselines import StaticPlanner, RandomPlanner, ReactivePlanner, StaticHighRiskPlannerV2
from planners.graphnet import GraphNetPlanner 

def run_batch_simulation(strategy_name, model_path=None, n_games=5000, chunk_size=500, device="cuda"):
    """
    运行批量模拟并返回每轮的平均合作率 (加入了防 OOM 的分块机制)
    """
    # 1. 选择并初始化策略 (Planner 只需要初始化一次)
    if strategy_name == "static": 
        planner = StaticPlanner()
    elif strategy_name == "random": 
        planner = RandomPlanner()
    elif strategy_name == "reactive": 
        planner = ReactivePlanner()
    
    elif strategy_name == "statichigh": 
        planner = StaticHighRiskPlannerV2()

    elif strategy_name == "graphnet":

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"[错误] 未找到模型: {model_path}")
        planner = GraphNetPlanner(model_path=model_path, device=device)
    else: 
        raise ValueError("Unknown strategy")

    # 用于累加所有 Chunk 的合作率
    accumulated_coop_rates = np.zeros(GameConfig.EPISODE_LENGTH)
    
    # 计算需要分多少个 Chunk
    num_chunks = int(np.ceil(n_games / chunk_size))
    print(f"  └─ 将 {n_games} 局分为 {num_chunks} 个 Chunk 进行计算 (Chunk Size: {chunk_size})...")

    # 2. 分块运行博弈
    for chunk_idx in range(num_chunks):
        # 计算当前 chunk 实际包含的局数 (处理不能整除的情况)
        current_batch_size = min(chunk_size, n_games - chunk_idx * chunk_size)
        
        # 初始化当前 chunk 的环境
        env = PublicGoodsGame(batch_size=current_batch_size, device=device)
        
        chunk_coop_rates = []

        with torch.no_grad():
            # Reset (执行 Round 1)
            capital, prev_decisions, edge_features = env.reset()
            chunk_coop_rates.append(prev_decisions.float().mean().item())

            # Round 2 - 15
            for t in range(GameConfig.EPISODE_LENGTH - 1):
                logits = planner.get_logits(capital, prev_decisions, edge_features, t + 1)
                next_state, _, _, _ = env.step(logits)
                capital, prev_decisions, edge_features = next_state
                
                chunk_coop_rates.append(prev_decisions.float().mean().item())
        
        # 按照当前 chunk 的规模加权累加到总和中
        accumulated_coop_rates += np.array(chunk_coop_rates) * current_batch_size
        
        # 显式清理当前 chunk 的显存，防止 OOM 累积
        del env, capital, prev_decisions, edge_features
        torch.cuda.empty_cache()
        gc.collect()

    # 计算全局平均合作率
    final_avg_coop_rates = accumulated_coop_rates / n_games
    return final_avg_coop_rates.tolist()

def plot_results(n_games, model_path, chunk_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[系统] 正在使用 {device} 运行 {n_games} 局平均测试...")

    strategies = ["static", "statichigh", "random", "reactive", "graphnet"]
    colors = {"static": "#95a5a6", "statichigh": "#e74c3c", "random": "#f1c40f", "reactive": "#e67e22", "graphnet": "#27ae60"}
    labels = {"static": "Static", "statichigh": "Static High Risk", "random": "Random", "reactive": "Reactive", "graphnet": "GraphNet Agent"}

    plt.figure(figsize=(10, 6))
    
    for str_name in strategies:
        print(f"\n▶ 正在运行 {str_name.upper()} 策略 ...")
        data = run_batch_simulation(
            strategy_name=str_name, 
            model_path=model_path, 
            n_games=n_games, 
            chunk_size=chunk_size, 
            device=device
        )
        
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
    # 如果是在服务器终端运行，通常不需要 plt.show()，注释掉防止卡住
    # plt.show() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000, help="运行的总游戏局数")
    parser.add_argument("--chunk_size", type=int, default=500, help="单次放入GPU运行的局数(防OOM)")
    parser.add_argument("--model_path", type=str, 
                        default="checkpoints/replicate_0/final_model.pth",
                        help="GraphNet模型路径")
    args = parser.parse_args()
    
    plot_results(n_games=args.n, model_path=args.model_path, chunk_size=args.chunk_size)