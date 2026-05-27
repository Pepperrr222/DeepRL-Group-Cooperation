# del2.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from env.game import PublicGoodsGame_v2 
from config import GameConfig, MODE, BotConfig
from planners.baselines import StaticPlannerV2, RandomPlannerV2, ReactivePlannerV2, StaticHighRiskPlannerV2

# 强制确保环境为 V2 模式
assert MODE == 1, "请在 config.py 中设置 MODE = 1"

def run_simulation(env, planner, n_games, delta):
    """运行一局完整的批量游戏并返回每轮合作率"""
    coop_rates = []
    
    # Reset (Round 1)
    capital, prev_decisions, edge_features = env.reset()
    coop_rates.append(prev_decisions.float().mean().item())
    
    # 模拟剩下的 14 轮
    with torch.no_grad():
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            # 获取决策 Logits
            logits = planner.get_logits(capital, prev_decisions, edge_features, t + 1)
            
            # 传入 delta 到 step 函数
            next_state, _, _, _ = env.step(logits, delta=delta)
            capital, prev_decisions, edge_features = next_state
            
            # 记录平均合作率
            coop_rates.append(prev_decisions.float().mean().item())
            
    return coop_rates

def main():
    # 检查 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_games = 1000
    
    # --- 实验设置 ---
    # delta 从 1 到 10，步长 0.1 (使用 round 处理浮点数精度问题)
    deltas = [round(d, 1) for d in np.arange(1.0, 10.1, 0.1)]
    # B_HIGH 扫描 0.1 到 0.8
    b_highs = [round(x * 0.1, 1) for x in range(1, 9)]
    # 固定 C_HIGH
    fixed_c_high = 0.1
    
    base_dir = "test_bh_delta_scan"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    print(f"🚀 开始大规模扫描实验...")
    print(f"📊 B_HIGH: {len(b_highs)} 档 | Delta: {len(deltas)} 档 | 总组合: {len(b_highs)*len(deltas)}")
    print(f"💻 使用设备: {device}")

    for b_val in b_highs:
        # 1. 为每个 B_HIGH 创建子文件夹
        sub_dir = os.path.join(base_dir, f"BHIGH_{b_val}")
        os.makedirs(sub_dir, exist_ok=True)
        
        # 2. 动态更新收益矩阵配置
        GameConfig.C_HIGH = fixed_c_high
        GameConfig.B_HIGH = b_val
        # 注意：这里需要根据具体的 C_HIGH/B_HIGH 更新矩阵，假设 C 为固定值或比例
        # 这里统一使用 config 中定义的 C_HIGH 和 C_LOW 逻辑
        c_low = GameConfig.C_LOW
        b_low = GameConfig.B_LOW
        
        GameConfig.LOW_RISK_MATRIX = [[0.0, b_low], [-c_low, b_low - c_low]]
        GameConfig.HIGH_RISK_MATRIX = [[0.0, b_val], [-fixed_c_high, b_val - fixed_c_high]]
        
        # 初始化 Planner 实例 (在 delta 循环外初始化以节省开销)
        strategies = {
            "Static": StaticPlannerV2(),
            "Random": RandomPlannerV2(),
            "Reactive": ReactivePlannerV2(),
            "StaticHighRisk": StaticHighRiskPlannerV2()
        }
        
        for d_val in deltas:
            # 3. 初始化环境
            env = PublicGoodsGame_v2(batch_size=n_games, device=device)
            
            results = {}
            for name, planner in strategies.items():
                results[name] = run_simulation(env, planner, n_games, delta=d_val)

            # 4. 绘图
            plt.figure(figsize=(10, 7))
            rounds = np.arange(1, GameConfig.EPISODE_LENGTH + 1)
            
            plt.plot(rounds, results["Static"], 'g--', label='Static (Low)', marker='o', markersize=4)
            plt.plot(rounds, results["Random"], 'r-', label='Random (30% High)', marker='s', markersize=4)
            plt.plot(rounds, results["Reactive"], 'b-', label='Reactive (C,C->High)', marker='^', markersize=4)
            plt.plot(rounds, results["StaticHighRisk"], 'm-', label='Static High Risk', marker='d', markersize=4)
            
            plt.title(f"Coop Rate Scan\nB_HIGH={b_val}, Delta={d_val}, C_HIGH={fixed_c_high}")
            plt.xlabel("Round")
            plt.ylabel("Avg Cooperation Rate")
            plt.ylim(-0.05, 1.05)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            
            # 保存图片，文件名格式化为 delta_1.5.png
            save_path = os.path.join(sub_dir, f"delta_{d_val}.png")
            plt.savefig(save_path)
            plt.close() # 释放内存
            
        print(f"✅ 完成 B_HIGH={b_val} 下的所有 Delta 测试")

    print(f"\n✨ 任务全部完成！结果已存入 {base_dir} 文件夹。")

if __name__ == "__main__":
    main()