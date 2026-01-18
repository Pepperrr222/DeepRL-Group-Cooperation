import torch
import numpy as np
import pandas as pd
import math
from env.game import PublicGoodsGame
from config import GameConfig

class RandomPlanner:
    """
    论文中的 Random Recommendations 基线：
    "recommends changing each edge with 30% probability"
    """
    def __init__(self, device):
        self.device = device
        # 预先计算 Logits，使得 Softmax 后 P(Keep)=0.7, P(Change)=0.3
        # Logits = [ln(0.7), ln(0.3)]
        # Softmax([ln(a), ln(b)]) = [a/(a+b), b/(a+b)]
        self.p_change = 0.3
        self.p_keep = 1.0 - self.p_change
        
        # 构造 logits: [Keep, Change]
        self.fixed_logits_val = torch.tensor(
            [math.log(self.p_keep), math.log(self.p_change)], 
            device=device
        )

    def get_action_logits(self, batch_size, n_players):
        """
        生成固定的概率分布 Logits。
        Shape: (Batch, N, N, 2)
        """
        # 1. 创建形状 (B, N, N, 2)
        logits = self.fixed_logits_val.view(1, 1, 1, 2).expand(batch_size, n_players, n_players, -1)
        return logits

def run_random_baseline(num_games=100000):
    device = torch.device("cpu") # 模拟通常不需要 GPU
    print(f"\n[系统] 初始化 {num_games} 局并行游戏 (Random Recommendations Baseline)...")
    print(f"[策略] 随机改变概率: 30% (Add if not exists, Delete if exists)")
    
    # 1. 初始化
    env = PublicGoodsGame(batch_size=num_games, device=device)
    planner = RandomPlanner(device)
    
    # 2. Reset (Round 1)
    capital, prev_decisions, adj = env.reset()
    
    stats = []
    
    # 记录第1轮
    stats.append({
        "Round": 1,
        "Coop Rate": prev_decisions.float().mean().item(),
        "Avg Capital": capital.mean().item(),
        "Avg Edges": adj.sum().item() / 2 / num_games
    })
    
    # 3. 循环 Round 2 - 15
    with torch.no_grad():
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            current_round = t + 2
            
            # A. 生成随机建议 Logits
            # 这里的 Logits 会导致 env.step 内部采样时，有 30% 的几率选中 1 (Change)
            logits = planner.get_action_logits(num_games, GameConfig.N_PLAYERS)
            
            # B. 环境交互
            # env.step 会根据 logits 采样动作，并根据当前图的状态自动转换：
            # - 如果采样到 Change(1) 且当前无边 -> 建议 Add
            # - 如果采样到 Change(1) 且当前有边 -> 建议 Delete
            next_state, _, _, actions_change = env.step(logits)
            capital, prev_decisions, adj = next_state
            
            # 统计建议数量 (调试用)
            # 理论上应该接近 Total_Edges * 0.3
            actual_change_rate = actions_change.float().mean().item()
            
            # C. 记录数据
            stats.append({
                "Round": current_round,
                "Coop Rate": prev_decisions.float().mean().item(),
                "Avg Capital": capital.mean().item(),
                "Avg Edges": adj.sum().item() / 2 / num_games,
                "Planner Activity": actual_change_rate # 应该在 0.3 左右
            })

    # 4. 输出结果
    df = pd.DataFrame(stats)
    
    print("\n" + "="*70)
    print(f"📊 100局游戏平均数据 (Random Planner | P=0.3)")
    print("="*70)
    print(df.to_string(index=False, formatters={
        "Coop Rate": "{:.2%}".format,
        "Avg Capital": "${:.2f}".format,
        "Avg Edges": "{:.1f}".format,
        "Planner Activity": "{:.1%}".format
    }))
    print("-" * 70)
    
    # 5. 结论对比
    final_coop = df.iloc[-1]["Coop Rate"]
    print(f"📉 最终合作率: {final_coop:.2%}")
    print("\n[论文对比参考]")
    print("- Rand et al. (2011): 11轮后约 60%")
    print("- Shirado et al. (2013): 15轮后约 40%")
    print("注意: 模拟结果取决于 Bot 的参数拟合程度，通常会介于静态网络(Static)和智能规划(AI)之间。")

if __name__ == "__main__":
    try:
        run_random_baseline(100)
    except ImportError:
        print("[错误] 请确保安装了 pandas: pip install pandas")
    except Exception as e:
        print(f"[错误] {e}")