import matplotlib.pyplot as plt
import numpy as np
import sys

# 导入你提供的 benchmark 模块
try:
    from ave import run_simulation
    from config import GameConfig
except ImportError as e:
    print(f"[错误] 无法导入模块: {e}")
    sys.exit(1)

# ================= 配置 =================
# 背景里画多少条细线 (模拟单局)
N_BACKGROUND_TRACES = 15 
# 均值线用多少局数据来计算 (越多越平滑准确)
N_MEAN_CALCULATION = 15

ROUNDS = np.arange(1, GameConfig.EPISODE_LENGTH + 1)

# 颜色配置 (保持论文风格)
STYLES = {
    "static":           {"title": "a  Static network",          "c_line": "#FFC0CB", "c_mean": "#DC143C"}, # 粉/红
    "random":           {"title": "b  Random recommendations",  "c_line": "#F0E68C", "c_mean": "#B8860B"}, # 黄/棕
    "coop_clustering":  {"title": "c  Cooperative clustering",  "c_line": "#90EE90", "c_mean": "#228B22"}, # 浅绿/深绿
    "graphnet":         {"title": "d  GraphNet planner",        "c_line": "#ADD8E6", "c_mean": "#1E90FF"}  # 浅蓝/深蓝
}

def get_plotting_data(strategy):
    """
    获取画图所需的数据：
    1. 多条单局轨迹 (用于背景)
    2. 一条高精度平均轨迹 (用于前景)
    """
    print(f"正在处理策略: [{strategy}] ...")
    
    # 1. 获取高精度均值 (利用 run_simulation 的并行能力)
    # 这会返回 (avg_coop, avg_cap)，我们只需要 coop (索引0)
    mean_coop_curve, _ = run_simulation(strategy, total_games=N_MEAN_CALCULATION, batch_size=2000)
    
    if mean_coop_curve is None:
        return None, None

    # 2. 获取单局轨迹 (用于画背景细线)
    # 因为 run_simulation 内部做了平均，为了拿单局数据，我们强制 batch=1
    individual_traces = []
    for _ in range(N_BACKGROUND_TRACES):
        # 跑 1 局
        single_coop, _ = run_simulation(strategy, total_games=1, batch_size=1)
        individual_traces.append(single_coop)
    
    return np.array(individual_traces), mean_coop_curve

def plot_cooperation_curves():
    strategies = ["static", "random", "coop_clustering", "graphnet"]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True, dpi=120)
    plt.subplots_adjust(wspace=0.05)

    for i, strat in enumerate(strategies):
        ax = axes[i]
        style = STYLES[strat]
        
        # 调用新的数据获取函数
        traces, mean_curve = get_plotting_data(strat)
        
        if mean_curve is None:
            ax.text(0.5, 0.5, "Model Not Found", ha='center')
            continue

        # 1. 画单局细线 (Background)
        for trace in traces:
            ax.plot(ROUNDS, trace, color=style['c_line'], alpha=0.4, linewidth=1)
            
        # 2. 画高精度均值虚线 (Foreground)
        # 注意：这里直接使用算好的 mean_curve，而不是对 traces 求均值
        # 这样结果更具有统计学意义
        ax.plot(ROUNDS, mean_curve, color=style['c_mean'], linestyle='--', linewidth=2.5, label='Mean')

        # 3. 样式调整
        ax.set_title(style['title'], loc='left', fontweight='bold', fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(1, 15)
        ax.set_xticks([1, 5, 10, 15])
        ax.set_xlabel("Round")
        
        if i == 0:
            ax.set_ylabel("Fraction of group cooperating")
        
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    output_file = "fig2_coop_curves.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\n 图片已保存: {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_cooperation_curves()