# sin2.py
import os
import torch
import numpy as np
from env.game import PublicGoodsGame
from config import GameConfig, MODE
from planners.baselines import StaticPlanner, RandomPlanner, ReactivePlanner
from planners.graphnet import GraphNetPlanner # 假设你封装了加载模型的类

try:
    from env.llm_bots import LLMBots
except ImportError:
    LLMBots = None

def get_gini(x):
    """计算基尼系数"""
    if x.sum() == 0: return 0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))

def run_single_game(strategy_name="graphnet", use_llm=False, llm_kwargs=None):
    device = torch.device("cpu")
    
    # 1. 初始化环境
    env = PublicGoodsGame(batch_size=1, device=device)

    # 如果启用 LLM，替换环境中的 bots
    if use_llm:
        if LLMBots is None:
            raise ImportError("未找到 env.llm_bots 或缺少依赖(openai 库)。请 pip install openai")
        if llm_kwargs is None or not llm_kwargs.get("api_key"):
            raise ValueError("使用 LLM 必须提供 api_key。")
        print(f"[系统] 正在将模拟 Bot 替换为 LLM ({llm_kwargs.get('model')})...")
        env.bots = LLMBots(
            batch_size=1,
            device=device,
            api_key=llm_kwargs["api_key"],
            base_url=llm_kwargs.get("base_url"),
            model=llm_kwargs.get("model")
        )

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
    parser.add_argument("--use_llm", action="store_true", help="启用 LLM 替代模拟 Bot")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="API Key")
    parser.add_argument("--base_url", type=str, default=None, help="自定义 API 地址 (DeepSeek/通义等)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="模型名称")
    args = parser.parse_args()

    if args.use_llm and not args.api_key:
        print("\n[错误] 开启 --use_llm 但未提供 api_key。")
        print("请添加 --api_key sk-xxx 或设置环境变量 OPENAI_API_KEY")
        import sys; sys.exit(1)

    llm_kwargs = {
        "api_key": args.api_key,
        "base_url": args.base_url,
        "model": args.model
    } if args.use_llm else None

    run_single_game(args.strategy, use_llm=args.use_llm, llm_kwargs=llm_kwargs)