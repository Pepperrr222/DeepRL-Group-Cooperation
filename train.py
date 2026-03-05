# train.py
import argparse
import random
import numpy as np
import torch
import multiprocessing as mp
from config import TrainConfig
from training.trainer import Trainer

def run_replicate(agent_id, base_seed):
    """单副本训练进程函数"""
    # 为每个进程设置唯一的种子
    seed = base_seed + agent_id
    
    # 重新初始化种子（在多进程环境下非常重要）
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 初始化训练器并开始
    trainer = Trainer(agent_id=agent_id, seed=seed)
    try:
        trainer.train()
    except Exception as e:
        print(f"Error in replicate {agent_id}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=30, help="要训练的独立副本数量")
    parser.add_argument("--parallel", type=int, default=1, help="同时并行的任务数(取决于你的GPU/显存)")
    args = parser.parse_args()

    print(f"Preparing to train {args.replicates} replicates...")

    # 如果 parallel=1, 则顺序执行；如果 >1, 使用进程池
    if args.parallel > 1:
        # 使用多进程池并行执行
        # 注意：如果你的 Agent 运行在 GPU 上，设置 parallel 过高会导致 OOM (显存溢出)
        # 建议 parallel 数 = 显存 / (单个Agent占用显存)
        ctx = mp.get_context('spawn') # GPU 任务推荐使用 spawn
        with ctx.Pool(args.parallel) as pool:
            pool.starmap(run_replicate, [(i, args.seed) for i in range(args.replicates)])
    else:
        # 顺序执行（适合显存不足的情况）
        for i in range(args.replicates):
            run_replicate(i, args.seed)

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()