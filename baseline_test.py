import torch
import numpy as np
import pandas as pd
from env.game import PublicGoodsGame
from config import GameConfig

def run_static_baseline(num_games=100):
    """
    运行静态网络基线测试 (No Agent Intervention)
    """
    device = torch.device("cpu")
    print(f"\n[系统] 正在初始化 {num_games} 局并行游戏 (Static Network Baseline)...")
    
    # 1. 初始化环境 (Batch Size = 100)
    # 这样我们可以同时模拟 100 个平行的平行宇宙
    env = PublicGoodsGame(batch_size=num_games, device=device)
    
    # 2. Reset (运行 Round 1)
    capital, prev_decisions, adj = env.reset()
    
    # 用于记录每一轮的统计数据 (平均值)
    stats = []
    
    # 记录第1轮数据
    stats.append({
        "Round": 1,
        "Coop Rate": prev_decisions.float().mean().item(),
        "Avg Capital": capital.mean().item(),
        "Avg Edges": adj.sum().item() / 2 / num_games # 除以2(无向图) 除以局数
    })
    
    # 3. 构造一个 "什么都不做" 的 Logits
    # 我们需要构造一个 Agent 输出，使得网络 100% 选择 "Keep" (Index 0)
    # Logits: [High_Value, Low_Value] -> Softmax -> [1.0, 0.0]
    # Shape: (Batch, N, N, 2)
    fake_logits = torch.zeros(num_games, GameConfig.N_PLAYERS, GameConfig.N_PLAYERS, 2, device=device)
    fake_logits[..., 0] = 100.0  # Index 0: Keep (保持不变)
    fake_logits[..., 1] = -100.0 # Index 1: Change (改变)
    
    # 4. 循环 Round 2 - 15
    # 不需要计算梯度，纯模拟
    with torch.no_grad():
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            current_round = t + 2
            
            # Step: 传入全是 "Keep" 的建议
            # 实际上这会导致 actions_change 全为 0，rec_type 全为 0
            # Bot 没有收到任何建议，网络结构将保持不变
            next_state, _, _, _ = env.step(fake_logits)
            capital, prev_decisions, adj = next_state
            
            # 记录数据
            stats.append({
                "Round": current_round,
                "Coop Rate": prev_decisions.float().mean().item(),
                "Avg Capital": capital.mean().item(),
                "Avg Edges": adj.sum().item() / 2 / num_games
            })

    # 5. 输出结果表格
    df = pd.DataFrame(stats)
    
    print("\n" + "="*60)
    print(f"📊 100局游戏平均数据 (无Agent干预/静态网络)")
    print("="*60)
    # 格式化打印
    print(df.to_string(index=False, formatters={
        "Coop Rate": "{:.2%}".format,
        "Avg Capital": "${:.2f}".format,
        "Avg Edges": "{:.1f}".format
    }))
    print("-" * 60)
    
    # 6. 最终总结
    final_coop = df.iloc[-1]["Coop Rate"]
    final_cap = df.iloc[-1]["Avg Capital"]
    start_coop = df.iloc[0]["Coop Rate"]
    
    print(f"📉 合作率变化: {start_coop:.2%} -> {final_coop:.2%}")
    print(f"💰 最终平均资金: ${final_cap:.2f}")
    
    if final_coop < start_coop:
        print("\n[结论] 符合公地悲剧：在没有外部干预的情况下，合作率随时间下降。")
    else:
        print("\n[结论] 异常：合作率未下降 (检查 Bot 参数是否过于友善)。")

if __name__ == "__main__":
    # 需要安装 pandas: pip install pandas
    try:
        run_static_baseline(100)
    except Exception as e:
        print(e)