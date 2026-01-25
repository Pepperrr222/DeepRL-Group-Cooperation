import torch
import numpy as np
import argparse
import sys
import os

# 导入环境
try:
    from env.game import PublicGoodsGame
    from config import GameConfig
except ImportError as e:
    print(f"[错误] 无法导入环境模块: {e}")
    sys.exit(1)

# 导入 Planners
try:
    from planners import (
        StaticPlanner, RandomPlanner, MaxConnectivityPlanner,
        CoopClusteringPlanner, EncouragementPlanner, NeutralPlanner,
        GraphNetPlanner
    )
except ImportError as e:
    print(f"[错误] 无法导入 Planner 模块。请确保 'planners' 文件夹存在。详情: {e}")
    sys.exit(1)

def get_planner(strategy, device):
    """策略工厂函数"""
    if strategy == "static": return StaticPlanner()
    if strategy == "random": return RandomPlanner()
    if strategy == "max_connectivity": return MaxConnectivityPlanner()
    if strategy == "coop_clustering": return CoopClusteringPlanner()
    if strategy == "encouragement": return EncouragementPlanner()
    if strategy == "neutral": return NeutralPlanner()
    if strategy == "graphnet": 
        # 默认寻找当前目录下的 checkpoints
        return GraphNetPlanner("checkpoints/final_model.pth", device)
    raise ValueError(f"Unknown strategy: {strategy}")

def run_simulation(strategy, device="cpu"):
    """
    执行一次完整的游戏模拟。
    
    Args:
        strategy (str): 策略名称
        device (str): 计算设备 'cpu' 或 'cuda'
        
    Returns:
        dict: 包含以下 Key 的数据字典:
            - 'adjacency': List[np.ndarray] (15个 (16,16) 矩阵, 0/1, 可完全还原图)
            - 'cooperation': List[float] (15个数值, 0~1)
            - 'avg_capital': List[float] (15个数值)
            - 'total_capital': List[float] (15个数值, 用于脚本打印)
    """
    if isinstance(device, str):
        device = torch.device(device)

    # 1. 初始化
    try:
        planner = get_planner(strategy, device)
    except Exception as e:
        # 为了防止在调用时崩溃，这里抛出运行时错误
        raise RuntimeError(f"Planner初始化失败: {e}")

    env = PublicGoodsGame(batch_size=1, device=device)
    
    # 数据容器
    history = {
        'adjacency': [],
        'cooperation': [],
        'avg_capital': [],
        'total_capital': []
    }

    def record_state(adj, decisions, capital):
        """辅助函数：记录当前帧状态"""
        # 转为 numpy 并 copy，防止引用覆盖
        history['adjacency'].append(adj[0].cpu().numpy().copy())
        history['cooperation'].append(decisions[0].float().mean().item())
        history['avg_capital'].append(capital[0].mean().item())
        history['total_capital'].append(capital[0].sum().item())

    # 2. Reset (Round 1)
    capital, prev_decisions, adj = env.reset()
    record_state(adj, prev_decisions, capital)
    
    # 3. Loop (Round 2-15)
    with torch.no_grad():
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            
            # 决策
            logits = planner.get_logits(capital, prev_decisions, adj, t + 1)
            
            # 环境执行
            next_state, _, _, _ = env.step(logits)
            capital, prev_decisions, adj = next_state
            
            # 记录数据
            record_state(adj, prev_decisions, capital)

    return history

if __name__ == "__main__":
    # --- 命令行入口逻辑 ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="graphnet", 
                        choices=["static", "random", "graphnet", "coop_clustering", 
                                 "encouragement", "neutral", "max_connectivity"],
                        help="选择使用的策略")
    args = parser.parse_args()
    
    print(f"正在运行策略: [{args.strategy.upper()}] ...")
    
    try:
        # 调用函数
        data = run_simulation(args.strategy)
        
        # 打印表格表头
        print("\n" + "="*55)
        print(f"{'Round':^6} | {'Coop Rate':^12} | {'Avg Capital':^15} | {'Total Capital':^15}")
        print("-" * 55)
        
        # 遍历打印每一轮
        num_rounds = len(data['cooperation'])
        for i in range(num_rounds):
            r_num = i + 1
            coop = data['cooperation'][i]
            avg_cap = data['avg_capital'][i]
            tot_cap = data['total_capital'][i]
            
            # 格式化输出
            print(f"{r_num:^6d} | {coop:^12.2%} | {avg_cap:^15.2f} | {tot_cap:^15.2f}")
            
        print("="*55)
        print(" 运行结束")
        
    except Exception as e:
        print(f"\n 运行出错: {e}")