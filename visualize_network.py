import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

from src.environment.game_env import NetworkGameEnv
from src.agents.human_bots import HumanBot
from src.planner.gnn_model import SocialPlannerAgent
from src.planner.policy import SocialPlannerPolicy
from src.training.trainer import SocialPlannerTrainer

def visualize_game_evolution(model_path="saved_models/social_planner_final.pth"):
    # 1. 加载环境和模型
    num_players = 16
    trainer = SocialPlannerTrainer(num_players=num_players)
    
    # 加载训练好的参数
    if os.path.exists(model_path):
        trainer.planner.load_state_dict(torch.load(model_path))
        print(f"✅ 成功加载模型: {model_path}")
    else:
        print("⚠️ 未找到模型文件，将使用随机初始化的模型进行演示...")

    trainer.planner.eval() # 切换到评估模式
    
    # 2. 运行一局游戏 (15 Rounds)
    env = NetworkGameEnv(num_players)
    bots = HumanBot(num_players)
    
    current_payoffs = np.zeros(num_players)
    last_actions = np.zeros(num_players)
    
    # 我们只画第 0, 5, 10, 14 回合的快照
    snapshots = [0, 5, 10, 14]
    
    plt.figure(figsize=(15, 4))
    
    for r in range(15):
        # 记录快照
        if r in snapshots:
            idx = snapshots.index(r)
            plt.subplot(1, 4, idx + 1)
            draw_network(env.graph, last_actions, r)

        # --- AI 介入流程 (同 Trainer) ---
        x, edge_attr, u = trainer.feature_adapter(
            env.adj_matrix, last_actions, current_payoffs, r, 15
        )
        edge_logits, _ = trainer.planner(x, edge_attr, u)
        proposed_adj_tensor, _, _ = trainer.policy.get_action(edge_logits, deterministic=True)
        proposed_adj = proposed_adj_tensor.squeeze(0).cpu().detach().numpy()
        
        # Bots 决定是否接受
        current_adj = env.adj_matrix
        final_adj = current_adj.copy()
        for i in range(num_players):
            for j in range(i + 1, num_players):
                if proposed_adj[i][j] != current_adj[i][j]:
                    action_type = 1 if proposed_adj[i][j] == 1 else -1
                    if bots.decide_acceptance(i, j, action_type, last_actions[j]) and \
                       bots.decide_acceptance(j, i, action_type, last_actions[i]):
                        final_adj[i][j] = final_adj[j][i] = proposed_adj[i][j]
        
        env.update_graph(final_adj)
        actions = bots.decide_cooperation(env.adj_matrix, r)
        current_payoffs = env.calculate_payoffs(actions)
        last_actions = actions

    plt.tight_layout()
    plt.savefig("network_evolution.png")
    print("🎨 网络演化图已保存为: network_evolution.png")
    # plt.show()

def draw_network(graph, actions, round_num):
    """画图辅助函数"""
    pos = nx.spring_layout(graph, seed=42) # 固定布局，方便对比
    
    # 颜色: 合作=蓝色, 背叛=红色
    colors = ['#4A90E2' if a == 1 else '#E74C3C' for a in actions]
    
    nx.draw(graph, pos, 
            node_color=colors, 
            with_labels=True, 
            node_size=300, 
            edge_color='gray', 
            width=0.5,
            alpha=0.8)
    
    plt.title(f"Round {round_num}")

if __name__ == "__main__":
    visualize_game_evolution()