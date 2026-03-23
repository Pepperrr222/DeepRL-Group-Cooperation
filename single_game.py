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

# 导入 LLM Bots (新增)
try:
    from env.llm_bots import LLMBots
except ImportError:
    LLMBots = None  # 允许在没有 openai 库时正常运行非 LLM 模式

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

def run_simulation(strategy, device="cpu", use_llm=False, llm_kwargs=None):
    """
    执行一次完整的游戏模拟。
    
    Args:
        strategy (str): 策略名称
        device (str): 计算设备 'cpu' 或 'cuda'
        use_llm (bool): 是否使用 LLM 代理替代数学公式 Bot
        llm_kwargs (dict): 传递给 LLMBots 的参数字典 (api_key, base_url, model)
        
    Returns:
        dict: 数据字典
    """
    if isinstance(device, str):
        device = torch.device(device)

    # 1. 初始化 Planner 和 环境
    try:
        planner = get_planner(strategy, device)
    except Exception as e:
        raise RuntimeError(f"Planner初始化失败: {e}")

    env = PublicGoodsGame(batch_size=1, device=device)
    
    # ==========================================
    # [核心修改]：如果启用了 LLM，动态替换环境里的 bots
    # ==========================================
    if use_llm:
        if LLMBots is None:
            raise ImportError("未找到 env.llm_bots 或缺少依赖(如 openai 库)。")
        if llm_kwargs is None or not llm_kwargs.get("api_key"):
            raise ValueError("使用 LLM 必须提供 api_key。")
            
        print(f"[系统] 🚀 正在将模拟人类替换为 LLM ({llm_kwargs.get('model')})...")
        env.bots = LLMBots(
            batch_size=1, 
            device=device, 
            api_key=llm_kwargs["api_key"],
            base_url=llm_kwargs.get("base_url"),
            model=llm_kwargs.get("model")
        )
    # ==========================================
    
    # 数据容器
    history = {
        'adjacency': [],
        'cooperation': [],
        'avg_capital': [],
        'total_capital':[]
    }

    def record_state(adj, decisions, capital):
        """辅助函数：记录当前帧状态"""
        history['adjacency'].append(adj[0].cpu().numpy().copy())
        history['cooperation'].append(decisions[0].float().mean().item())
        history['avg_capital'].append(capital[0].mean().item())
        history['total_capital'].append(capital[0].sum().item())

    # 2. Reset (Round 1) - LLM 会在这里进行第一次决策
    print("[系统] 初始化 Round 1...")
    capital, prev_decisions, adj = env.reset()
    record_state(adj, prev_decisions, capital)
    
    # 3. Loop (Round 2-15)
    with torch.no_grad():
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            if use_llm:
                print(f"[系统] 正在运行 Round {t + 2} (LLM 正在思考，可能需要一些时间)...")
                
            # 决策
            logits = planner.get_logits(capital, prev_decisions, adj, t + 1)
            
            # 环境执行 (包含了 LLM 处理 Agent建议 + 下一轮合作决策)
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
                        help="选择使用的规划器策略")

    # --- 新增: LLM 相关参数 ---
    parser.add_argument("--use_llm", action="store_true", help="启用后，将使用 LLM API 作为玩家")
    # 默认从环境变量读取 OPENAI_API_KEY，如果没有则为空字符串
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="API Key")
    parser.add_argument("--base_url", type=str, default=None, help="自定义 API 代理地址 (如 DeepSeek/阿里等)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="使用的大语言模型名称")
    # --------------------------

    args = parser.parse_args()
    
    # 校验 API Key
    if args.use_llm and not args.api_key:
        print("\n[错误] 您开启了 --use_llm，但未提供 API Key。")
        print("请在命令行添加 --api_key sk-xxx 或设置环境变量 OPENAI_API_KEY")
        sys.exit(1)
    
    print(f"正在运行策略: [{args.strategy.upper()}] ...")
    
    try:
        # 打包 LLM 参数
        llm_kwargs = {
            "api_key": args.api_key,
            "base_url": args.base_url,
            "model": args.model
        }
        
        # 调用函数
        data = run_simulation(
            strategy=args.strategy, 
            use_llm=args.use_llm, 
            llm_kwargs=llm_kwargs
        )
        
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
        import traceback
        traceback.print_exc()