import torch
import numpy as np
import os
import sys

# 尝试导入项目模块
try:
    from model.agent import SocialPlannerAgent
    from env.game import PublicGoodsGame
    from config import GameConfig, TrainConfig
except ImportError as e:
    print(f"[错误] 无法导入模块: {e}")
    sys.exit(1)

def evaluate_100_games_fast():
    # 1. 配置
    # 纯数据计算，使用 GPU (如果可用) 会非常快
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "checkpoints/ckpt_ep_158900.pth"
    total_games = 10000
    
    print(f"\n[系统] 设备: {device}")
    print(f"[系统] 正在初始化 {total_games} 局并行游戏...")

    # 2. 加载模型
    agent = SocialPlannerAgent().to(device)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        agent.load_state_dict(state_dict)
        agent.eval()
        print(f"[系统] 成功加载模型: {model_path}")
    else:
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先运行 python train.py 进行训练。")
        return

    # 3. 初始化并行环境
    # batch_size=100 意味着我们同时模拟 100 个平行宇宙
    env = PublicGoodsGame(batch_size=total_games, device=device)
    
    # 4. 收集数据的容器
    # shape: (Rounds, Games)
    history_coop = np.zeros((GameConfig.EPISODE_LENGTH, total_games))
    history_cap = np.zeros((GameConfig.EPISODE_LENGTH, total_games))
    history_edges = np.zeros((GameConfig.EPISODE_LENGTH, total_games))

    # 5. 开始模拟
    # --- Round 1 (Reset) ---
    capital, prev_decisions, adj = env.reset()
    
    # 记录 Round 1 数据
    history_coop[0] = prev_decisions.float().mean(dim=1).cpu().numpy()
    history_cap[0] = capital.mean(dim=1).cpu().numpy()
    # 边数计算: sum / 2 (无向图)
    history_edges[0] = adj.sum(dim=(1, 2)).float().cpu().numpy() / 2.0

    print("[系统] 开始推理 (Round 2-15)...")
    
    with torch.no_grad():
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            current_round_idx = t + 1 # Index 1 to 14 (Round 2 to 15)
            
            # Agent 决策
            edge_logits, _ = agent(capital, prev_decisions, adj, current_round_idx)
            
            # 环境交互
            next_state, _, _, _ = env.step(edge_logits)
            capital, prev_decisions, adj = next_state
            
            # 记录数据
            history_coop[current_round_idx] = prev_decisions.float().mean(dim=1).cpu().numpy()
            history_cap[current_round_idx] = capital.mean(dim=1).cpu().numpy()
            history_edges[current_round_idx] = adj.sum(dim=(1, 2)).float().cpu().numpy() / 2.0

    # 6. 统计分析
    print("\n" + "="*65)
    print(f"📊 100 局游戏统计汇总 (Trained Agent)")
    print("="*65)
    print(f"{'Round':^6} | {'Coop Rate (Avg)':^18} | {'Capital (Avg)':^15} | {'Edges (Avg)':^12}")
    print("-" * 65)

    for r in range(GameConfig.EPISODE_LENGTH):
        avg_c = np.mean(history_coop[r])
        std_c = np.std(history_coop[r])
        avg_m = np.mean(history_cap[r])
        avg_e = np.mean(history_edges[r])
        
        # 打印每一轮的平均数据
        print(f" {r+1:02d}    | {avg_c:6.2%} (±{std_c:.2f}) | ${avg_m:6.2f}       | {avg_e:6.1f}")

    print("-" * 65)
    
    # 7. 最终结论
    final_coop_avg = np.mean(history_coop[-1])
    final_coop_std = np.std(history_coop[-1])
    final_cap_avg = np.mean(history_cap[-1])
    
    print(f"\n📈 最终结果摘要:")
    print(f"  - 平均最终合作率: {final_coop_avg:.2%} (标准差: {final_coop_std:.2f})")
    print(f"  - 平均最终资金  : ${final_cap_avg:.2f}")
 

    for label, count in zip(labels, counts):
        bar = "█" * int(count / 2)
        print(f"  {label:8s}: {count:3d} 局 | {bar}")

    # 判断效果
    start_coop = np.mean(history_coop[0])


if __name__ == "__main__":
    evaluate_100_games_fast()