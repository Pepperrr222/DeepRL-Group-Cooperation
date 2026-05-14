# tra2.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse
def plot_training_history(rep=0):
    replicate_id = rep
    # 查找指定 replicate 的日志文件
    log_path = f"checkpoints/replicate_{replicate_id}/training_log.csv"
    
    if not os.path.exists(log_path):
        print(f"[错误] 未找到日志文件: {log_path}")
        return

    # 读取数据
    df = pd.read_csv(log_path)
    
    # 绘图
    plt.figure(figsize=(10, 5))
    
    # 绘制合作率曲线
    plt.plot(df['episode'], df['coop_rate'], label='Coop Rate', color='#3498db', alpha=0.7)
    
    # 简单的移动平均线，防止曲线过于抖动
    if len(df) > 10:
        ma = df['coop_rate'].rolling(window=10).mean()
        plt.plot(df['episode'], ma, color='#2c3e50', label='Moving Avg (10 steps)')
    
    plt.title(f"Replicate {replicate_id} Training History")
    plt.xlabel("Episode (Steps)")
    plt.ylabel("Cooperation Rate")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 保存图片
    save_path = f"training_curve_rep_{replicate_id}.png"
    plt.savefig(save_path)
    print(f"✅ 训练曲线已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, default=0, help="replicate ID")
    args = parser.parse_args()


    plot_training_history(rep=args.rep)

    