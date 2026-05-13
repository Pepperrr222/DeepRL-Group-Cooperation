# run_experiments.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from env.game import PublicGoodsGame_v2
from config import GameConfig, BotConfig, MODE
from planners.baselines import StaticPlannerV2, RandomPlannerV2, ReactivePlannerV2

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
            
            # 环境 Step
            # 注意：我们需要给 bots 传 delta。
            # 这里我们通过修改 env 内部的 bots 默认 delta 参数来实现
            next_state, _, _, _ = env.step(logits, delta=delta)
            capital, prev_decisions, edge_features = next_state
            
            # 记录平均合作率
            coop_rates.append(prev_decisions.float().mean().item())
            
    return coop_rates

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_games = 1000
    
    # 实验范围
    deltas = list(range(1, 11)) 
    c_highs = [round(x * 0.1, 1) for x in range(1, 9)] # 0.1 到 0.8
    
    base_dir = "test3lenth15"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    print(f"开始实验：共 {len(c_highs) * len(deltas)} 种组合...")

    for c_val in c_highs:
        # 1. 为每个 C_HIGH 创建文件夹
        sub_dir = os.path.join(base_dir, f"CHIGH_{c_val}")
        os.makedirs(sub_dir, exist_ok=True)
        
        # 2. 动态更新全局配置中的收益矩阵
        GameConfig.C_HIGH = c_val
        GameConfig.HIGH_RISK_MATRIX = [
            [0.0, GameConfig.B_HIGH],
            [-GameConfig.C_HIGH, GameConfig.B_HIGH - GameConfig.C_HIGH]
        ]
        
        for d_val in deltas:
            print(f"正在测试: C_HIGH={c_val}, Delta={d_val} ...")
            
            # 3. 初始化环境（此时环境会读取最新的 GameConfig）
            env = PublicGoodsGame_v2(batch_size=n_games, device=device)
            
            # 为了能传 delta 到底层，我们对 env.step 做一个小包装或直接修改 Bot 属性
            # 这里简单起见，我们直接给 env.bots 注入一个属性
            env.step_original = env.step
            def patched_step(action_logits, delta=d_val):
                # 重新绑定底层调用，强制闭包 delta
                B, N = env.bs, env.n
                # 这部分需要稍微改动 env/game.py 让它接受 delta 参数传给 bots
                # 如果没改 game.py，可以在这里通过 hack 方式修改 bots 的方法属性
                return env.step_original(action_logits, delta=delta)
            
            # 我们直接手动运行三种策略
            results = {}
            strategies = {
                "Static": StaticPlannerV2(),
                "Random": RandomPlannerV2(),
                "Reactive": ReactivePlannerV2()
            }
            
            for name, planner in strategies.items():
                # 注意：这里需要确保你的 game.py 的 step 函数已经改成了可以接受 delta 参数
                # 如果你还没改 game.py 的 step 签名，请告诉我，我给你 Patch
                res = run_simulation(env, planner, n_games, delta=d_val)
                results[name] = res

            # 4. 绘图
            plt.figure(figsize=(8, 6))
            rounds = np.arange(1, GameConfig.EPISODE_LENGTH + 1)
            
            plt.plot(rounds, results["Static"], 'g--', label='Static (Low Risk)', marker='o')
            plt.plot(rounds, results["Random"], 'r-', label='Random (30% High)', marker='s')
            plt.plot(rounds, results["Reactive"], 'b-', label='Reactive (C,C->High)', marker='^')
            
            plt.title(f"Cooperation Rate (C_HIGH={c_val}, Delta={d_val})")
            plt.xlabel("Round")
            plt.ylabel("Avg Cooperation Rate")
            plt.ylim(0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 保存
            save_path = os.path.join(sub_dir, f"delta_{d_val}.png")
            plt.savefig(save_path)
            plt.close()

    print(f"\n✅ 所有实验完成！结果已存入 {base_dir} 文件夹。")

if __name__ == "__main__":
    main()