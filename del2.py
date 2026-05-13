#del2.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from env.game import PublicGoodsGame_v2 # 显式引用 V2 环境
from config import GameConfig, MODE
from planners.baselines import StaticPlannerV2, RandomPlannerV2, ReactivePlannerV2

# 强制确保环境为 V2 模式
assert MODE == 1, "请在 config.py 中设置 MODE = 1"

def run_simulation(env, planner, n_games, delta):
    """运行一局完整的批量游戏并返回每轮合作率"""
    coop_rates =[]
    
    # Reset (Round 1)
    capital, prev_decisions, edge_features = env.reset()
    coop_rates.append(prev_decisions.float().mean().item())
    
    # 模拟剩下的 14 轮
    with torch.no_grad():
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            # 获取决策 Logits
            logits = planner.get_logits(capital, prev_decisions, edge_features, t + 1)
            
            # 【关键修改】：将 delta 传入 step 函数
            next_state, _, _, _ = env.step(logits, delta=delta)
            capital, prev_decisions, edge_features = next_state
            
            # 记录平均合作率
            coop_rates.append(prev_decisions.float().mean().item())
            
    return coop_rates

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_games = 1000
    
    # --- 实验设置：C_HIGH 固定为 0.5，B_HIGH 扫描 0.1 到 0.8 ---
    deltas = [1, 5, 10] 
    b_highs = [round(x * 0.1, 1) for x in range(1, 9)]
    fixed_c_high = 0.1
    
    base_dir = "test_bh_scan"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    print(f"🚀 开始 B_HIGH 扫描实验：共 {len(b_highs) * len(deltas)} 种组合...")

    for b_val in b_highs:
        # 1. 创建子文件夹
        sub_dir = os.path.join(base_dir, f"BHIGH_{b_val}")
        os.makedirs(sub_dir, exist_ok=True)
        
        # 2. 动态更新收益矩阵
        GameConfig.C_HIGH = fixed_c_high
        GameConfig.B_HIGH = b_val
        GameConfig.HIGH_RISK_MATRIX = [
            [0.0, GameConfig.B_HIGH],[-GameConfig.C_HIGH, GameConfig.B_HIGH - GameConfig.C_HIGH]
        ]
        
        for d_val in deltas:
            print(f"正在测试: B_HIGH={b_val}, Delta={d_val} ...")
            
            # 3. 初始化环境
            env = PublicGoodsGame_v2(batch_size=n_games, device=device)
            
            results = {}
            strategies = {
                "Static": StaticPlannerV2(),
                "Random": RandomPlannerV2(),
                "Reactive": ReactivePlannerV2()
            }
            
            for name, planner in strategies.items():
                results[name] = run_simulation(env, planner, n_games, delta=d_val)

            # 4. 绘图
            plt.figure(figsize=(8, 6))
            rounds = np.arange(1, GameConfig.EPISODE_LENGTH + 1)
            
            plt.plot(rounds, results["Static"], 'g--', label='Static (Low)', marker='o')
            plt.plot(rounds, results["Random"], 'r-', label='Random (30% High)', marker='s')
            plt.plot(rounds, results["Reactive"], 'b-', label='Reactive (C,C->High)', marker='^')
            
            plt.title(f"Cooperation Rate (B_HIGH={b_val}, Delta={d_val}, C_HIGH={fixed_c_high})")
            plt.xlabel("Round")
            plt.ylabel("Avg Cooperation Rate")
            plt.ylim(0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.savefig(os.path.join(sub_dir, f"delta_{d_val}.png"))
            plt.close()

    print(f"\n✅ 所有实验完成！结果已存入 {base_dir} 文件夹。")

if __name__ == "__main__":
    main()