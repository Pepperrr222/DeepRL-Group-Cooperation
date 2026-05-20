#coo22.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import gc
import networkx as nx
from env.game import PublicGoodsGame
from config import GameConfig, MODE
from planners.baselines import StaticPlanner, RandomPlanner, ReactivePlanner
from planners.graphnet import GraphNetPlanner 

# 确保在机制设计模式下运行
assert MODE == 1, "[错误] 该脚本专为 V2 机制设计模式编写，请确保 config.py 中 MODE = 1"

def inject_graph_topology(env, graph_type):
    """
    动态替换环境中的图池，生成不同拓扑结构的网络
    保证网络的平均度 (Average Degree) 约等于 config 中的目标值 (默认 4)
    """
    n = GameConfig.N_PLAYERS
    degree = int(getattr(GameConfig, 'TARGET_AVG_DEGREE', 4))
    
    for i in range(env.pool_size):
        if graph_type == "RRG":
            # 随机规则图: 所有节点度数严格为 degree
            G = nx.random_regular_graph(d=degree, n=n)
        elif graph_type == "BA":
            # 无标度网络 (Barabasi-Albert): 具有大V节点 (Hubs)
            G = nx.barabasi_albert_graph(n=n, m=degree//2)
        elif graph_type == "WS":
            # 小世界网络 (Watts-Strogatz): 高聚类系数，少量随机重连
            G = nx.watts_strogatz_graph(n=n, k=degree, p=0.1)
        elif graph_type == "ER":
            # 经典随机图 (Erdos-Renyi)
            p = degree / (n - 1)
            G = nx.erdos_renyi_graph(n=n, p=p)
        else:
            raise ValueError(f"未知的图拓扑类型: {graph_type}")
            
        # 替换环境内部的图池
        env.rrg_pool[i] = torch.tensor(nx.to_numpy_array(G), dtype=torch.float, device=env.device)

def run_batch_simulation(strategy_name, graph_type, model_path=None, n_games=5000, chunk_size=500, device="cuda"):
    """
    运行指定策略和拓扑的批量模拟
    """
    # 1. 选择 Planner
    if strategy_name == "static": planner = StaticPlanner()
    elif strategy_name == "random": planner = RandomPlanner()
    elif strategy_name == "reactive": planner = ReactivePlanner()
    elif strategy_name == "graphnet":
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"[错误] 未找到模型: {model_path}")
        planner = GraphNetPlanner(model_path=model_path, device=device)
    else: raise ValueError("Unknown strategy")

    accumulated_coop_rates = np.zeros(GameConfig.EPISODE_LENGTH)
    num_chunks = int(np.ceil(n_games / chunk_size))

    # 2. 分块运行
    for chunk_idx in range(num_chunks):
        current_batch_size = min(chunk_size, n_games - chunk_idx * chunk_size)
        
        # 初始化环境
        env = PublicGoodsGame(batch_size=current_batch_size, device=device)
        
        # --- 核心操作：注入特定的网络拓扑 ---
        inject_graph_topology(env, graph_type)
        
        chunk_coop_rates = []

        with torch.no_grad():
            # Reset (Round 1)
            capital, prev_decisions, edge_features = env.reset()
            chunk_coop_rates.append(prev_decisions.float().mean().item())

            # Round 2 - 15
            for t in range(GameConfig.EPISODE_LENGTH - 1):
                logits = planner.get_logits(capital, prev_decisions, edge_features, t + 1)
                next_state, _, _, _ = env.step(logits)
                capital, prev_decisions, edge_features = next_state
                
                chunk_coop_rates.append(prev_decisions.float().mean().item())
        
        # 加权累加
        accumulated_coop_rates += np.array(chunk_coop_rates) * current_batch_size
        
        # 释放显存
        del env, capital, prev_decisions, edge_features
        torch.cuda.empty_cache()
        gc.collect()

    final_avg_coop_rates = accumulated_coop_rates / n_games
    return final_avg_coop_rates.tolist()

def plot_results(n_games, model_path, chunk_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[系统] 使用 {device} 运行多拓扑测试 (局数: {n_games}/图)...")

    topologies = ["RRG", "BA", "WS", "ER"]
    strategies = ["static", "random", "reactive", "graphnet"]
    
    colors = {"static": "#95a5a6", "random": "#f1c40f", "reactive": "#e67e22", "graphnet": "#27ae60"}
    labels = {"static": "Static", "random": "Random", "reactive": "Reactive", "graphnet": "GraphNet Agent"}
    topo_titles = {"RRG": "Random Regular Graph", "BA": "Scale-Free (Barabasi-Albert)", 
                   "WS": "Small-World (Watts-Strogatz)", "ER": "Erdos-Renyi Random Graph"}

    # 创建输出文件夹
    out_dir = "topology_comparisons"
    os.makedirs(out_dir, exist_ok=True)

    # 外层循环：遍历 4 种拓扑结构
    for topo in topologies:
        print(f"\n" + "="*50)
        print(f"🌍 开始测试网络拓扑: {topo_titles[topo]}")
        print("="*50)
        
        plt.figure(figsize=(9, 6))
        
        # 内层循环：遍历 4 种策略
        for str_name in strategies:
            print(f"  ▶ 正在运行 {str_name.upper()} ...")
            data = run_batch_simulation(
                strategy_name=str_name, 
                graph_type=topo,          # 传入当前拓扑
                model_path=model_path, 
                n_games=n_games, 
                chunk_size=chunk_size, 
                device=device
            )
            
            rounds = np.arange(1, GameConfig.EPISODE_LENGTH + 1)
            plt.plot(rounds, data, label=labels[str_name], 
                     color=colors[str_name], marker='o', markersize=5, linewidth=2)

        # 当前拓扑图表的独立美化
        plt.title(f"Cooperation Rate Evolution | Topology: {topo_titles[topo]}\n(Avg over {n_games} Games, Mode: V2)", fontsize=13)
        plt.xlabel("Round", fontsize=12)
        plt.ylabel("Average Cooperation Rate", fontsize=12)
        plt.ylim(0, 1.0)
        plt.xticks(np.arange(1, 16))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best')
        
        # 保存独立图片
        filename = os.path.join(out_dir, f"coop_rate_{topo}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close() # 关闭当前图表，准备画下一张
        
        print(f"  ✅ 拓扑 {topo} 测试完毕，已保存为: {filename}")

    print(f"\n🎉 所有拓扑测试已完成！4 张对比图已保存在 '{out_dir}' 目录下。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="运行的总游戏局数 (每种拓扑)")
    parser.add_argument("--chunk_size", type=int, default=500, help="单次放入GPU的局数")
    parser.add_argument("--model_path", type=str, 
                        default="checkpoints/replicate_0/final_model.pth",
                        help="GraphNet模型路径")
    args = parser.parse_args()
    
    plot_results(n_games=args.n, model_path=args.model_path, chunk_size=args.chunk_size)