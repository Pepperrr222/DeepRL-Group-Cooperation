import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from src.environment.game_env import NetworkGameEnv
from src.training.trainer import SocialPlannerTrainer
from src.agents.llm_bots import LLMBot

# ================= 配置区域 =================
MOCK_MODE = False  # <--- 先设置为 True 进行测试！跑通后再改为 False

# 当 MOCK_MODE = False 时，下面这些才生效
API_KEY = "sk-aonzxraxsctwtfshddtbaytnqpikuwssvhendbhhizohiaol" 
BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
# ===========================================

def main():
    print(f"🚀 启动实验 (模式: {'Mock/模拟' if MOCK_MODE else 'Real/真实LLM'})")
    
    # 1. 加载 Planner
    trainer = SocialPlannerTrainer(num_players=16)
    model_path = "saved_models/social_planner_final.pth"
    
    if os.path.exists(model_path):
        trainer.planner.load_state_dict(torch.load(model_path))
        trainer.planner.eval()
        print(f"✅ AI Planner 模型加载成功")
    else:
        print("❌ 警告：未找到模型文件！将使用随机初始化的 Planner 进行演示。")

    # 2. 初始化环境和 Bots
    env = NetworkGameEnv(num_players=16)
    
    # 注意这里传入了 mock 参数
    bots = LLMBot(num_players=16, 
                  api_key=API_KEY if not MOCK_MODE else "dummy", 
                  base_url=BASE_URL, 
                  model_name=MODEL_NAME,
                  mock=MOCK_MODE)
    
    # 3. 跑 3 轮测试
    MAX_ROUNDS = 10
    history_coop = []
    
    current_payoffs = np.zeros(16)
    last_actions = np.zeros(16)

    for r in range(MAX_ROUNDS):
        print(f"\n{'='*10} Round {r+1} {'='*10}")
        
        # --- A. Planner 建议 ---
        x, edge_attr, u = trainer.feature_adapter(
            env.adj_matrix, last_actions, current_payoffs, r, MAX_ROUNDS
        )
        edge_logits, _ = trainer.planner(x, edge_attr, u)
        proposed_adj_tensor, _, _ = trainer.policy.get_action(edge_logits, deterministic=True)
        proposed_adj = proposed_adj_tensor.squeeze(0).cpu().detach().numpy()
        
        # --- B. Bot 接受/拒绝 ---
        current_adj = env.adj_matrix
        final_adj = current_adj.copy()
        changes = 0
        
        for i in range(16):
            for j in range(i + 1, 16):
                if proposed_adj[i][j] != current_adj[i][j]:
                    action_type = 1 if proposed_adj[i][j] == 1 else -1
                    if bots.decide_acceptance(i, j, action_type, last_actions[j]) and \
                       bots.decide_acceptance(j, i, action_type, last_actions[i]):
                        final_adj[i][j] = final_adj[j][i] = proposed_adj[i][j]
                        changes += 1
        
        print(f"✅ 网络变动: {changes} 处修改")
        env.update_graph(final_adj)
        
        # --- C. Bot 决策 ---
        actions = bots.decide_cooperation(env.adj_matrix, r)
        
        # --- D. 统计 ---
        coop_rate = np.mean(actions)
        history_coop.append(coop_rate)
        current_payoffs = env.calculate_payoffs(actions)
        last_actions = actions
        
        print(f"📊 合作率: {coop_rate:.2%}")

    # 4. 保存结果图
    plt.plot(range(1, MAX_ROUNDS+1), history_coop, marker='o')
    plt.title(f"Test Run (Mock={MOCK_MODE})")
    plt.savefig("test_llm_run.png")
    print("\n✅ 测试通过！结果图已保存为 test_llm_run.png")

if __name__ == "__main__":
    main()