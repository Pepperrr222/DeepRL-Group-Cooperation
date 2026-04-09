import torch
import numpy as np
import argparse
import sys
from tqdm import tqdm

# 导入项目模块
try:
    from env.game import PublicGoodsGame
    from config import GameConfig
    from planners import (
        StaticPlanner, RandomPlanner, MaxConnectivityPlanner,
        CoopClusteringPlanner, EncouragementPlanner, NeutralPlanner,
        GraphNetPlanner
    )
except ImportError as e:
    print(f"[错误] 无法导入模块: {e}")
    sys.exit(1)

def get_planner(strategy, device):
    """策略工厂"""
    if strategy == "static": 
        return StaticPlanner()
    if strategy == "random": 
        return RandomPlanner()
    if strategy == "max_connectivity": 
        return MaxConnectivityPlanner()
    if strategy == "coop_clustering": 
        return CoopClusteringPlanner()
    if strategy == "encouragement": 
        return EncouragementPlanner()
    if strategy == "neutral": 
        return NeutralPlanner()
    if strategy == "graphnet": 
        return GraphNetPlanner("checkpoints/final_model.pth", device)
    raise ValueError(f"Unknown strategy: {strategy}")

def run_simulation(strategy, total_games=1000000, batch_size=2000):
    """
    运行大规模基准测试并返回统计数据。
    
    Returns:
        tuple: (avg_coop_per_round, avg_cap_per_round)
            - avg_coop_per_round: np.array, shape=(15,), 每回合平均合作率
            - avg_cap_per_round: np.array, shape=(15,), 每回合平均资金
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 仅在作为脚本运行时打印系统信息，避免调用时刷屏
    if __name__ == "__main__":
        print(f"\n[系统] 设备: {device}")
        print(f"[任务] 策略: {strategy.upper()} | 总局数: {total_games} | Batch: {batch_size}")

    # 1. 初始化 Planner
    try:
        planner = get_planner(strategy, device)
    except Exception as e:
        print(f"[错误] {e}")
        return None, None

    # 计算需要跑多少个 Batch
    num_batches = int(np.ceil(total_games / batch_size))
    
    # 累加器 (Accumulators)
    global_coop_sum = torch.zeros(GameConfig.EPISODE_LENGTH, device=device)
    global_cap_sum = torch.zeros(GameConfig.EPISODE_LENGTH, device=device)
    
    actual_total_games = 0

    # 2. 批量循环
    # 如果是被调用，disable=True 可以关闭进度条，或者根据需求保留
    show_progress = (__name__ == "__main__")
    iterator = tqdm(range(num_batches), desc="Benchmarking", disable=not show_progress)
    
    for _ in iterator:
        current_bs = min(batch_size, total_games - actual_total_games)
        if current_bs <= 0: break
        
        env = PublicGoodsGame(batch_size=current_bs, device=device)
        
        # --- Round 1 (Reset) ---
        capital, prev_decisions, adj = env.reset()
        
        global_coop_sum[0] += prev_decisions.float().mean(dim=1).sum()
        global_cap_sum[0] += capital.mean(dim=1).sum()
        
        # --- Loop Round 2-15 ---
        with torch.no_grad():
            for t in range(GameConfig.EPISODE_LENGTH - 1):
                current_round_idx = t + 1
                
                logits = planner.get_logits(capital, prev_decisions, adj, current_round_idx)
                
                next_state, _, _, _ = env.step(logits)
                capital, prev_decisions, adj = next_state
                
                global_coop_sum[current_round_idx] += prev_decisions.float().mean(dim=1).sum()
                global_cap_sum[current_round_idx] += capital.mean(dim=1).sum()
        
        actual_total_games += current_bs

    # 3. 计算平均值
    avg_coop_per_round = (global_coop_sum / actual_total_games).cpu().numpy()
    avg_cap_per_round = (global_cap_sum / actual_total_games).cpu().numpy()

    return avg_coop_per_round, avg_cap_per_round

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="graphnet", 
                        choices=["static", "random", "graphnet", "coop_clustering", 
                                 "encouragement", "neutral", "max_connectivity"])
    parser.add_argument("--total", type=int, default=100000, help="总游戏次数")
    parser.add_argument("--batch", type=int, default=2000, help="单次并行数量")
    
    args = parser.parse_args()
    
    # 调用函数获取数据
    coop_data, cap_data = run_simulation(args.strategy, args.total, args.batch)
    
    if coop_data is not None:
        # 4. 打印逻辑 (保留原有效果)
        print("\n" + "="*55)
        print(f" 测试结果 (N={args.total})")
        print(f"   策略: {args.strategy.upper()}")
        print("="*55)
        print(f"{'Round':^6} | {'Avg Coop Rate':^15} | {'Avg Capital':^15}")
        print("-" * 55)
        
        for r in range(GameConfig.EPISODE_LENGTH):
            print(f"{r+1:^6d} | {coop_data[r]:^15.2%} | {cap_data[r]:^15.2f}")
        
        print("-" * 55)
        print(f" 最终回合平均合作率: {coop_data[-1]:.2%}")
        print(f" 最终回合平均资金  : {cap_data[-1]:.2f}")
        print("="*55)