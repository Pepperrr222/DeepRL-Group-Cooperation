import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys

# 尝试导入项目模块
try:
    from model.agent import SocialPlannerAgent
    from env.game import PublicGoodsGame
    from config import GameConfig
except ImportError as e:
    print(f"[错误] 无法导入模块: {e}")
    sys.exit(1)

# ==========================================
# 1. 数据生成部分
# ==========================================

def get_coop_rates(strategy_name, num_games=20, model_path=None):
    """
    运行指定策略 num_games 次，返回合作率矩阵。
    Return shape: (num_games, num_rounds)
    """
    device = torch.device("cpu") # 绘图数据生成不需要 GPU 加速
    env = PublicGoodsGame(batch_size=num_games, device=device)
    
    # 结果容器: (Games, Rounds)
    coop_history = np.zeros((num_games, GameConfig.EPISODE_LENGTH))
    
    # --- Reset (Round 1) ---
    capital, prev_decisions, adj = env.reset()
    coop_history[:, 0] = prev_decisions.float().mean(dim=1).numpy()
    
    # 加载模型 (如果是 GraphNet)
    agent = None
    if strategy_name == "GraphNet":
        if not os.path.exists(model_path):
            print(f"[跳过] 找不到模型文件 {model_path}，无法生成 GraphNet 数据。")
            return None
        agent = SocialPlannerAgent().to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()
    
    # --- Loop Round 2-15 ---
    with torch.no_grad():
        for t in range(GameConfig.EPISODE_LENGTH - 1):
            current_round_idx = t + 1
            
            # --- 策略分支 ---
            if strategy_name == "Static":
                # 静态: 100% 保持现状 (Index 0 = Keep)
                logits = torch.zeros(num_games, GameConfig.N_PLAYERS, GameConfig.N_PLAYERS, 2)
                logits[..., 0] = 100.0 # Keep
                logits[..., 1] = -100.0
                
            elif strategy_name == "Random":
                # 随机: 30% 改变 (Change), 70% 保持 (Keep)
                # 构造 Logits 使得 Softmax 后为 [0.7, 0.3]
                p_change = 0.000001
                val_keep = math.log(1 - p_change)
                val_change = math.log(p_change)
                logits = torch.tensor([val_keep, val_change]).view(1, 1, 1, 2).expand(num_games, GameConfig.N_PLAYERS, GameConfig.N_PLAYERS, -1)
                
            elif strategy_name == "GraphNet":
                # AI: 模型前向传播
                logits, _ = agent(capital, prev_decisions, adj, current_round_idx)
            
            # --- 环境步进 ---
            next_state, _, _, _ = env.step(logits)
            capital, prev_decisions, adj = next_state
            
            # 记录合作率
            coop_history[:, current_round_idx] = prev_decisions.float().mean(dim=1).numpy()
            
    return coop_history

# ==========================================
# 2. 绘图部分
# ==========================================

def plot_paper_style():
    # 配置
    N_GAMES = 24  # 模拟论文中的线条数量
    MODEL_PATH = "checkpoints/final_model.pth"
    
    print(f"正在生成数据 (每种策略 {N_GAMES} 局)...")
    
    # 获取数据
    data_static = get_coop_rates("Static", N_GAMES)
    data_random = get_coop_rates("Random", N_GAMES)
    data_graphnet = get_coop_rates("GraphNet", N_GAMES, MODEL_PATH)
    
    # 准备画布 (1行3列)
    # 论文有4张图，但我们只复现了 Static, Random, GraphNet。Cooperative Clustering 是基于规则的，暂略。
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, dpi=150)
    plt.subplots_adjust(wspace=0.05) # 减小图之间的间距
    
    rounds = np.arange(1, GameConfig.EPISODE_LENGTH + 1)
    
    # 定义样式配置
    # 格式: (Data, Title, Color_Light, Color_Dark)
    configs = [
        (data_static, "a  Static network", "#FFC0CB", "#DC143C"),       # 粉色/深红
        (data_random, "b  Random recommendations", "#F0E68C", "#B8860B"), # 卡其/暗金
        (data_graphnet, "d  GraphNet planner", "#B0E0E6", "#1E90FF")      # 淡蓝/宝蓝
    ]
    
    # 循环绘制
    for ax, (data, title, c_light, c_dark) in zip(axes, configs):
        if data is None:
            ax.text(0.5, 0.5, "No Data (Train First)", ha='center')
            continue
            
        # 1. 画细线 (Individual Sessions)
        # alpha=0.4 实现半透明效果
        for i in range(len(data)):
            ax.plot(rounds, data[i], color=c_light, alpha=0.5, linewidth=1.5)
            
        # 2. 画均值线 (Mean)
        # linestyle='--' 虚线
        mean_data = np.mean(data, axis=0)
        ax.plot(rounds, mean_data, color=c_dark, linestyle='--', linewidth=2.5, label='Mean')
        
        # 3. 设置坐标轴和样式
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(1, 15)
        ax.set_xticks([1, 5, 10, 15])
        
        # 标题左对齐，加粗
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=10)
        
        # 网格线 (可选，论文里好像没有明显的网格，或者很淡)
        # ax.grid(True, linestyle=':', alpha=0.3)
        
        # 边框美化 (类似 R 的 ggplot 风格)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.tick_params(direction='out')

    # 只在第一个图显示 Y 轴标签
    axes[0].set_ylabel("Fraction of group cooperating", fontsize=12)
    
    # 在所有图下方显示 X 轴标签
    for ax in axes:
        ax.set_xlabel("Round", fontsize=12)

    # 保存
    save_path = "reproduced_figure_2.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✅ 图片已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_paper_style()