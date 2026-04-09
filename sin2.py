# single_game.py
import torch
import numpy as np
from env.game import PublicGoodsGame
from config import GameConfig, MODE
from planners.baselines import StaticPlanner, RandomPlanner, ReactivePlanner
from planners.graphnet import GraphNetPlanner # 假设你封装了加载模型的类

def get_gini(x):
    """计算基尼系数"""
    if x.sum() == 0: return 0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))

def run_single_game(strategy_name="graphnet"):
    device = torch.device("cpu")
    
    # 1. 初始化环境
    env = PublicGoodsGame(batch_size=1, device=device)
    
    # 2. 根据名称选择 Planner
    if strategy_name == "static": planner = StaticPlanner()
    elif strategy_name == "random": planner = RandomPlanner()
    elif strategy_name == "reactive": planner = ReactivePlanner()
    else: planner = GraphNetPlanner(device=device) # 默认加载训练好的模型

    # 3. 开始游戏
    print(f"\n" + "="*80)
    print(f"🎮 运行一局游戏 | 模式: {'V2 (规则设计)' if MODE==1 else 'V1 (拓扑干预)'} | 策略: {strategy_name.upper()}")
    print("="*80)

    # Reset 环境
    capital, prev_decisions, edge_features = env.reset()
    
    # 初始统计
    if MODE == 1:
        adj = edge_features[0, ..., 0] # 拓扑是固定的
        total_possible = GameConfig.N_PLAYERS * (GameConfig.N_PLAYERS - 1) / 2
        initial_conn = adj.sum().item() / 2 / total_possible
        print(f"📈 初始网络连接率: {initial_conn:.2%}")
    
    # 表格头
    header = f"{'轮次':^4} | {'合作率':^8} | {'高风险边数(%)':^15} | {'均资':^7} | {'基尼系数':^6} | {'建议采纳':^6}"
    print(header)
    print("-" * len(header))

    # 游戏循环
    for t in range(GameConfig.EPISODE_LENGTH):
        # Round 1 的数据来自 reset，Round 2-15 来自 step
        if t > 0:
            # Agent 获取 Logits
            # 注意：V2 传 edge_features，V1 传 adj
            logits = planner.get_logits(capital, prev_decisions, edge_features, t)
            
            # 环境 Step
            next_state, reward, dist, actions_change = env.step(logits)
            capital, prev_decisions, edge_features = next_state
        
        # --- 指标计算 ---
        # 1. 合作率
        coop_rate = prev_decisions[0].mean().item()
        
        # 2. 机制指标 (仅限 V2)
        if MODE == 1:
            adj = edge_features[0, ..., 0]
            game_modes = edge_features[0, ..., 1]
            total_active_edges = adj.sum().item() / 2
            high_risk_edges = (game_modes * adj).sum().item() / 2
            hr_percent = (high_risk_edges / total_active_edges) if total_active_edges > 0 else 0
            risk_str = f"{int(high_risk_edges):3d} ({hr_percent:4.1%})"
        else:
            risk_str = "N/A (V1)"

        # 3. 财富指标
        avg_cap = capital[0].mean().item()
        gini = get_gini(capital[0].numpy())

        # 4. 打印当前行
        print(f"{t+1:^6d} | {coop_rate:8.1%} | {risk_str:^15} | {avg_cap:8.2f} | {gini:8.3f} | {'-' if t==0 else 'Running'}")

    print("="*80)
    print(f"🏁 游戏总结: 最终平均资金 ${avg_cap:.2f}, 最终合作率 {coop_rate:.1%}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="static", choices=["static", "random", "reactive", "graphnet"])
    args = parser.parse_args()
    
    run_single_game(args.strategy)