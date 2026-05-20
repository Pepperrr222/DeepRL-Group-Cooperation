# ave2.py
import torch
import numpy as np
import argparse
import os
import gc
from env.game import PublicGoodsGame
from config import GameConfig, MODE
from planners.baselines import StaticPlanner, RandomPlanner, ReactivePlanner
from planners.graphnet import GraphNetPlanner

def batch_gini(capital_matrix):
    """
    批量计算 Gini 系数。
    capital_matrix shape: (B, N)
    返回所有 B 局游戏中 Gini 系数的平均值。
    """
    # 按行(局)排序资金，防止出现极小负值导致错误，用 clamp 截断
    cap_clamped = torch.clamp(capital_matrix, min=0.0)
    sorted_cap, _ = torch.sort(cap_clamped, dim=1)
    B, N = sorted_cap.shape
    
    # 构造 index: 1 到 N
    index = torch.arange(1, N + 1, device=capital_matrix.device).float()
    
    # Gini = sum((2*i - n - 1) * x) / (n * sum(x))
    numerator = torch.sum((2 * index - N - 1) * sorted_cap, dim=1)
    denominator = N * torch.sum(sorted_cap, dim=1)
    
    # 避免除以 0
    gini_per_game = torch.where(denominator == 0, torch.zeros_like(numerator), numerator / denominator)
    return gini_per_game.mean().item()

def run_average_simulation(n_games=100, chunk_size=500, strategy_name="static", model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n" + "="*85)
    print(f" 批量运行 {n_games} 局游戏 | 模式: {'V2 (规则设计)' if MODE==1 else 'V1 (拓扑干预)'} | 策略: {strategy_name.upper()}")
    if strategy_name == "graphnet":
        print(f"📂 使用模型: {model_path}")
    print("="*85)
    
    # 1. 根据名称选择 Planner (Planner 无状态，全局复用一个即可)
    if strategy_name == "static": 
        planner = StaticPlanner()
    elif strategy_name == "random": 
        planner = RandomPlanner()
    elif strategy_name == "reactive": 
        planner = ReactivePlanner()
    elif strategy_name == "graphnet": 
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"[错误] 未找到模型文件: {model_path}，请先训练！")
        planner = GraphNetPlanner(model_path=model_path, device=device)
    else: 
        print(f"[警告] 未知策略 {strategy_name}，回退至 static")
        planner = StaticPlanner()

    # 2. 准备分块累加器 (记录每一轮的特征总和，最后统一除以 n_games)
    # 因为我们在循环外统计，因此需要把15轮的数据都存起来
    acc_coop_rate = np.zeros(GameConfig.EPISODE_LENGTH)
    acc_hr_edges = np.zeros(GameConfig.EPISODE_LENGTH)
    acc_hr_percent = np.zeros(GameConfig.EPISODE_LENGTH)
    acc_avg_cap = np.zeros(GameConfig.EPISODE_LENGTH)
    acc_avg_gini = np.zeros(GameConfig.EPISODE_LENGTH)
    
    num_chunks = int(np.ceil(n_games / chunk_size))
    print(f" 将 {n_games} 局任务切割为 {num_chunks} 个 Chunk 进行运算，单块最大 {chunk_size} 局...")
    
    avg_initial_conn = 0.0

    # 3. 游戏分块循环
    for chunk_idx in range(num_chunks):
        # 计算当前 chunk 的实际大小
        current_bs = min(chunk_size, n_games - chunk_idx * chunk_size)
        
        # 初始化当前 Chunk 的并行环境
        env = PublicGoodsGame(batch_size=current_bs, device=device)
        
        with torch.no_grad():
            # Reset 环境 (执行 Round 1)
            capital, prev_decisions, edge_features = env.reset()
            
            # 记录一次初始网络连接率
            if chunk_idx == 0 and MODE == 1:
                adj = edge_features[..., 0]
                total_possible = GameConfig.N_PLAYERS * (GameConfig.N_PLAYERS - 1) / 2
                avg_initial_conn = (adj.sum(dim=(1,2)) / 2).mean().item() / total_possible

            # 遍历 15 轮游戏
            for t in range(GameConfig.EPISODE_LENGTH):
                if t > 0:
                    logits = planner.get_logits(capital, prev_decisions, edge_features, t)
                    next_state, reward, dist, actions_change = env.step(logits)
                    capital, prev_decisions, edge_features = next_state
                
                # --- 累加当前批次的指标总和 ---
                
                # 合作率 (当前 Batch 的均值 * Batch大小 = 总合作率和)
                acc_coop_rate[t] += prev_decisions.float().mean(dim=1).sum().item()
                
                # 资金
                acc_avg_cap[t] += capital.mean(dim=1).sum().item()
                
                # 基尼系数 (batch_gini 返回均值，乘回 bs)
                acc_avg_gini[t] += batch_gini(capital) * current_bs
                
                # V2 特定指标
                if MODE == 1:
                    adj = edge_features[..., 0]        # (B, N, N)
                    game_modes = edge_features[..., 1] # (B, N, N)
                    
                    active_edges = adj.sum(dim=(1,2)) / 2 
                    high_risk_edges = (game_modes * adj).sum(dim=(1,2)) / 2
                    hr_percent = torch.where(active_edges > 0, high_risk_edges / active_edges, torch.zeros_like(active_edges))
                    
                    acc_hr_edges[t] += high_risk_edges.sum().item()
                    acc_hr_percent[t] += hr_percent.sum().item()

        # 释放显存，防止 OOM
        del env, capital, prev_decisions, edge_features
        torch.cuda.empty_cache()
        gc.collect()
        
        # 打印简单进度
        print(f"  └─ Chunk {chunk_idx + 1}/{num_chunks} 已完成 ({current_bs} 局)")

    # 4. 汇总与格式化输出
    if MODE == 1:
        print(f"\n {n_games}局平均初始网络连接率: {avg_initial_conn:.2%}")
        
    header = f"{'轮次':^4} | {'平均合作率':^10} | {'平均高风险边数(%)':^20} | {'平均资金':^10} | {'平均基尼系数':^10}"
    print(header)
    print("-" * len(header))

    final_coop_rate = 0.0
    final_avg_cap = 0.0

    for t in range(GameConfig.EPISODE_LENGTH):
        # 整体求平均
        coop_rate = acc_coop_rate[t] / n_games
        avg_cap = acc_avg_cap[t] / n_games
        avg_gini = acc_avg_gini[t] / n_games
        
        if MODE == 1:
            avg_hr_edges = acc_hr_edges[t] / n_games
            avg_hr_percent = acc_hr_percent[t] / n_games
            risk_str = f"{avg_hr_edges:5.1f} ({avg_hr_percent:5.1%})"
        else:
            risk_str = "N/A (V1)"

        if t == GameConfig.EPISODE_LENGTH - 1:
            final_coop_rate = coop_rate
            final_avg_cap = avg_cap

        print(f"{t+1:^6d} | {coop_rate:10.1%} | {risk_str:^22} | ${avg_cap:<9.2f} | {avg_gini:10.3f}")

    print("="*85)
    print(f" 综合总结 ({n_games} 局平均): 最终平均资金 ${final_avg_cap:.2f}, 最终合作率 {final_coop_rate:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="运行的游戏局数")
    # 【新增】支持手动调控分块大小，默认为 500
    parser.add_argument("--chunk_size", type=int, default=500, help="并行运算块大小(防OOM)")
    parser.add_argument("--strategy", type=str, default="static", choices=["static", "random", "reactive", "graphnet"])
    parser.add_argument("--model_path", type=str, default="checkpoints/replicate_0/final_model.pth", help="GraphNet 模型路径")
    args = parser.parse_args()
    
    run_average_simulation(
        n_games=args.n, 
        chunk_size=args.chunk_size,
        strategy_name=args.strategy, 
        model_path=args.model_path
    )