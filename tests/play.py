# play_demo.py
import torch
import os
import sys
import numpy as np

# 尝试导入项目模块
try:
    from model.agent import SocialPlannerAgent
    from env.game import PublicGoodsGame
    from config import GameConfig, TrainConfig
except ImportError as e:
    print(f"[错误] 无法导入模块。请确保你在项目根目录下运行。错误详情: {e}")
    sys.exit(1)

def print_player_status(decisions, capitals):
    """
    以双栏表格形式打印所有玩家的详细状态
    """
    N = len(decisions)
    mid = N // 2
    
    print(f"  👥 [玩家详细状态]")
    print(f"     ID | 行为 |  资金   ||   ID | 行为 |  资金")
    print(f"    {'='*22}||{'='*22}")
    
    for i in range(mid):
        j = i + mid
        
        # 左栏玩家 (0-7)
        act_i = "🔵合作" if decisions[i] == 1 else "🔴背叛"
        cap_i = f"{capitals[i]:5.2f}"
        
        # 右栏玩家 (8-15)
        act_j = "🔵合作" if decisions[j] == 1 else "🔴背叛"
        cap_j = f"{capitals[j]:5.2f}"
        
        print(f"     {i:02d} | {act_i} | {cap_i}   ||   {j:02d} | {act_j} | {cap_j}")
    print(f"    {'-'*46}")

def get_action_details(old_adj, new_adj, actions_change):
    """分析建议详情"""
    logs = []
    N = old_adj.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            if actions_change[i, j] == 1:
                is_connected = (old_adj[i, j] == 1)
                did_change = (old_adj[i, j] != new_adj[i, j])
                if not is_connected:
                    action_str = f"➕ 建议连线 ({i:2d}-{j:2d})"
                    result_str = "✅ 接受" if did_change else "❌ 拒绝"
                else:
                    action_str = f"✂️ 建议断连 ({i:2d}-{j:2d})"
                    result_str = "✅ 接受" if did_change else "❌ 拒绝"
                logs.append(f"{action_str} | {result_str}")
    return logs

def run_simulation():
    # 1. 设置
    # 演示时强制使用 CPU，防止显存问题或初始化卡顿
    device = torch.device("cpu")
    print(f"[系统] 使用设备: {device}")

    # 2. 加载模型
    model_path = "checkpoints/final_model.pth"
    agent = SocialPlannerAgent().to(device)
    
    if os.path.exists(model_path):
        print(f"[系统] 加载模型: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        agent.load_state_dict(state_dict)
        agent.eval()
    else:
        print(f"[警告] 未找到模型，将使用 **随机策略** 进行演示...")

    # 3. 初始化环境
    env = PublicGoodsGame(batch_size=1, device=device)
    
    print("\n" + "="*60)
    print("🎮 开始模拟 (16 Players, 15 Rounds)")
    print("="*60)

    # 4. Reset (Round 1)
    capital, prev_decisions, adj = env.reset()
    
    # --- 打印 Round 1 状态 ---
    print(f"\n[Round 01] (初始状态)")
    # 打印详细列表
    print_player_status(prev_decisions[0], capital[0])
    
    # 打印宏观统计
    round_1_edges = adj[0].sum().item() / 2
    print(f"  📊 宏观: 合作率 {prev_decisions[0].float().mean():.2%} | 连接数 {int(round_1_edges)}")

    # 5. 循环 Round 2-15
    with torch.no_grad():
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            current_round = t + 2
            print(f"\n" + "="*60)
            
            # --- Agent 决策 ---
            edge_logits, value_est = agent(capital, prev_decisions, adj, t + 1)
            
            # --- 环境交互 ---
            old_adj = adj.clone()
            next_state, reward, dist, actions_change = env.step(edge_logits)
            new_capital, new_decisions, new_adj = next_state
            
            # --- 打印本轮详细信息 ---
            print(f"[Round {current_round:02d}] Agent估值: {value_est[0].item():.2f}")
            
            # 1. 打印详细玩家列表
            print_player_status(new_decisions[0], new_capital[0])
            
            # 2. 打印 Agent 建议
            log_actions = get_action_details(old_adj[0], new_adj[0], actions_change[0])
            if len(log_actions) > 0:
                print(f"  📝 Agent 建议 ({len(log_actions)} 条):")
                # 最多显示 5 条，防止刷屏
                for log in log_actions[:5]:
                    print(f"     {log}")
                if len(log_actions) > 5:
                    print(f"     ... (还有 {len(log_actions)-5} 条)")
            else:
                print(f"  💤 Agent 保持沉默")

            # 3. 宏观统计
            cur_edges = new_adj[0].sum().item() / 2
            print(f"  📊 宏观: 合作率 {new_decisions[0].float().mean():.2%} | 连接数 {int(cur_edges)}")
            
            # 更新状态
            capital, prev_decisions, adj = next_state

    print("\n" + "="*60)
    print("🏁 演示结束")

if __name__ == "__main__":
    run_simulation()