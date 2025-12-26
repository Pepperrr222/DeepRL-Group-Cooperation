import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# 导入我们的训练器
from src.training.trainer import SocialPlannerTrainer

def main():
    # --- 1. 配置参数 ---
    MAX_EPISODES = 10000   # 训练多少局 (论文可能训练了几万局，演示用2000即可看到效果)
    PRINT_INTERVAL = 100   # 每隔多少局打印一次日志
    SAVE_PATH = "saved_models"
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    print(f"🚀 开始训练 Social Planner! 目标: {MAX_EPISODES} 局")
    print("-" * 50)
    
    # --- 2. 初始化训练器 ---
    trainer = SocialPlannerTrainer(num_players=16, lr=0.001)
    
    # 用于记录数据画图
    history = {
        "cooperation_rate": [],
        "reward": [],
        "loss": []
    }
    
    # --- 3. 训练循环 ---
    for episode in range(1, MAX_EPISODES + 1):
        # 运行一整局 (15 Rounds) 并更新模型
        metrics = trainer.run_episode(train=True)
        
        # 记录数据
        history["cooperation_rate"].append(metrics["mean_cooperation"])
        history["reward"].append(metrics["total_reward"])
        history["loss"].append(metrics["loss"])
        
        # 打印进度
        if episode % PRINT_INTERVAL == 0:
            # 计算最近50局的平均值，数据更平滑
            avg_coop = np.mean(history["cooperation_rate"][-PRINT_INTERVAL:])
            avg_rew = np.mean(history["reward"][-PRINT_INTERVAL:])
            print(f"Episode {episode}/{MAX_EPISODES} | "
                  f"Coop Rate: {avg_coop:.2%} | "  # 比如 45.00%
                  f"Avg Reward: {avg_rew:.4f} | "
                  f"Loss: {metrics['loss']:.4f}")

    # --- 4. 保存模型 ---
    model_path = os.path.join(SAVE_PATH, "social_planner_final.pth")
    torch.save(trainer.planner.state_dict(), model_path)
    print("-" * 50)
    print(f"✅ 训练完成! 模型已保存至: {model_path}")
    
    # --- 5. 可视化结果 ---
    plot_training_results(history)

def plot_training_results(history):
    """
    画出训练过程中的合作率和奖励变化曲线
    """
    episodes = range(1, len(history["cooperation_rate"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 图1: 合作率 (Cooperation Rate)
    plt.subplot(1, 2, 1)
    # 绘制原始数据 (半透明)
    plt.plot(episodes, history["cooperation_rate"], alpha=0.3, color='gray')
    # 绘制移动平均线 (平滑)
    window_size = 50
    smooth_coop = np.convolve(history["cooperation_rate"], np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size, len(history["cooperation_rate"]) + 1), smooth_coop, color='blue', linewidth=2, label='Moving Avg')
    
    plt.title("Cooperation Rate over Time")
    plt.xlabel("Episode")
    plt.ylabel("Cooperation Rate (0-1)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 图2: 总奖励 (Total Reward)
    plt.subplot(1, 2, 2)
    plt.plot(episodes, history["reward"], alpha=0.3, color='gray')
    smooth_rew = np.convolve(history["reward"], np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size, len(history["reward"]) + 1), smooth_rew, color='orange', linewidth=2, label='Moving Avg')
    
    plt.title("Group Total Reward over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图片
    plt.tight_layout()
    plt.savefig("training_curve.png")
    print("📈 训练曲线图已保存为: training_curve.png")
    # 如果是在本地运行，可以用 plt.show()
    # plt.show()

if __name__ == "__main__":
    main()