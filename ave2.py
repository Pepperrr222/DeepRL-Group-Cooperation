# ave2.py
import torch
import numpy as np
import argparse
import os
from env.game import PublicGoodsGame
from config import GameConfig, MODE
from planners.baselines import StaticPlanner, RandomPlanner, ReactivePlanner
from planners.graphnet import GraphNetPlanner # 引入 GraphNetPlanner

def batch_gini(capital_matrix):
    """
    批量计算 Gini 系数。
    capital_matrix shape: (B, N)
    返回所有 B 局游戏中 Gini 系数的平均值。
    """
    # 按行(局)排序资金，防止出现极小负值导致错误，用 clamp 截断
    sorted_cap, _ = torch.sort(torch.clamp(capital_matrix, min=0.0), dim=1)
    B, N = sorted_cap.shape
    
    # 构造 index: 1 到 N
    index = torch.arange(1, N + 1, device=capital_matrix.device).float()
    
    # Gini = sum((2*i - n - 1) * x) / (n * sum(x))
    numerator = torch.sum((2 * index - N - 1) * sorted_cap, dim=1)
    denominator = N * torch.sum(sorted_cap, dim=1)
    
    # 避免除以 0
    gini_per_game = torch.where(denominator == 0, torch.zeros_like(numerator), numerator / denominator)
    return gini_per_game.mean().item()

# 【修改1】增加 model_path 参数
def run_average_simulation(n_games=100, strategy_name="static", model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n" + "="*85)
    print(f"🚀 批量运行 {n_games} 局游戏 | 模式: {'V2 (规则设计)' if MODE==1 else 'V1 (拓扑干预)'} | 策略: {strategy_name.upper()}")
    if strategy_name == "graphnet":
        print(f"📂 使用模型: {model_path}")
    print("="*85)
    
    # 1. 初始化并行环境
    env = PublicGoodsGame(batch_size=n_games, device=device)
    
    # 2. 根据名称选择 Planner
    if strategy_name == "static": 
        planner = StaticPlanner()
    elif strategy_name == "random": 
        planner = RandomPlanner()
    elif strategy_name == "reactive": 
        planner = ReactivePlanner()
    elif strategy_name == "graphnet": 
        # 【修改2】正确传入 model_path 和 device
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"[错误] 未找到模型文件: {model_path}，请先训练！")
        planner = GraphNetPlanner(model_path=model_path, device=device)
    else: 
        print(f"[警告] 未知策略 {strategy_name}，回退至 static")
        planner = StaticPlanner()

    # 3. Reset 环境 (执行 Round 1)
    capital, prev_decisions, edge_features = env.reset()
    
    # 初始网络连接率统计
    if MODE == 1:
        adj = edge_features[..., 0] # (B, N, N)
        total_possible = GameConfig.N_PLAYERS * (GameConfig.N_PLAYERS - 1) / 2
        # 计算所有局的平均连接数
        avg_initial_conn = (adj.sum(dim=(1,2)) / 2).mean().item() / total_possible
        print(f"📈 100局平均初始网络连接率: {avg_initial_conn:.2%}")
    
    # 表格头
    header = f"{'轮次':^4} | {'平均合作率':^10} | {'平均高风险边数(%)':^20} | {'平均资金':^10} | {'平均基尼系数':^10}"
    print(header)
    print("-" * len(header))

    # 数据记录容器（用于最后总结）
    final_coop_rate = 0.0
    final_avg_cap = 0.0

    # 4. 游戏循环 (Round 1 to 15)
    # 【修改3】批量测试时必须加上 no_grad，否则跑1000局显存会爆炸
    with torch.no_grad():
        for t in range(GameConfig.EPISODE_LENGTH):
            # Round 1 (t=0) 的数据已经由 reset 产生，直接跳过 step
            if t > 0:
                # Agent 获取 Logits (支持批量 B)
                logits = planner.get_logits(capital, prev_decisions, edge_features, t)
                
                # 环境 Step (并行处理 B 局)
                next_state, reward, dist, actions_change = env.step(logits)
                capital, prev_decisions, edge_features = next_state
            
            # --- 批量指标计算 ---
            
            # 1. 合作率 (平均所有局、所有玩家)
            coop_rate = prev_decisions.float().mean().item()
            
            # 2. 机制指标 (仅限 V2)
            if MODE == 1:
                adj = edge_features[..., 0]        # (B, N, N)
                game_modes = edge_features[..., 1] # (B, N, N)
                
                # 每局的活跃边数
                active_edges_per_game = adj.sum(dim=(1,2)) / 2 
                # 每局的高风险边数
                high_risk_edges_per_game = (game_modes * adj).sum(dim=(1,2)) / 2
                
                # 每局的高风险比例 (避免除零)
                hr_percent_per_game = torch.where(
                    active_edges_per_game > 0, 
                    high_risk_edges_per_game / active_edges_per_game, 
                    torch.zeros_like(active_edges_per_game)
                )
                
                # 总体均值
                avg_hr_edges = high_risk_edges_per_game.mean().item()
                avg_hr_percent = hr_percent_per_game.mean().item()
                
                risk_str = f"{avg_hr_edges:5.1f} ({avg_hr_percent:5.1%})"
            else:
                risk_str = "N/A (V1)"

            # 3. 财富与公平指标
            avg_cap = capital.mean().item()
            avg_gini = batch_gini(capital)

            # 记录最后一步的数据用于总结
            if t == GameConfig.EPISODE_LENGTH - 1:
                final_coop_rate = coop_rate
                final_avg_cap = avg_cap

            # 4. 打印当前行
            print(f"{t+1:^6d} | {coop_rate:10.1%} | {risk_str:^22} | ${avg_cap:<9.2f} | {avg_gini:10.3f}")

    print("="*85)
    print(f"🏁 综合总结 ({n_games} 局平均): 最终平均资金 ${final_avg_cap:.2f}, 最终合作率 {final_coop_rate:.1%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="运行的游戏局数")
    parser.add_argument("--strategy", type=str, default="static", choices=["static", "random", "reactive", "graphnet"])
    # 【修改4】增加命令行参数接收模型路径
    parser.add_argument("--model_path", type=str, default="checkpoints/replicate_0/final_model.pth", help="GraphNet 模型路径")
    args = parser.parse_args()
    
    # 传入 model_path
    run_average_simulation(n_games=args.n, strategy_name=args.strategy, model_path=args.model_path)