# train.py
import argparse
import time
import random
import numpy as np
import torch
import sys
import os

# 导入配置和训练器
from config import TrainConfig
from training.trainer import Trainer

def set_seed(seed):
    """
    设置全局随机种子，确保实验的可复现性。
    包括 Python random, NumPy, 和 PyTorch (CPU & GPU)。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 为了保证绝对的一致性，禁用 CUDNN 的 benchmark 模式
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Info] Random Seed set to: {seed}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Deep RL for Scaffolding Cooperation")
    
    # 基础参数
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for initialization (default: 42)")
    parser.add_argument("--device", type=str, default=None, 
                        help="Compute device to use (cpu or cuda)")
    
    # 训练参数覆盖 (覆盖 config.py 中的默认值)
    parser.add_argument("--episodes", type=int, default=None, 
                        help="Override total number of training episodes")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, 
                        help="Override learning rate")
    
    return parser.parse_args()

def main():
    # 1. 解析参数
    args = parse_args()

    # 2. 应用配置
    set_seed(args.seed)
    
    if args.device:
        TrainConfig.DEVICE = args.device
    
    if args.episodes:
        TrainConfig.MAX_EPISODES = args.episodes
        
    if args.batch_size:
        TrainConfig.BATCH_SIZE = args.batch_size
        
    if args.lr:
        TrainConfig.LR = args.lr

    # 3. 打印实验环境信息
    print("\n" + "="*50)
    print(f"   AI Social Planner - Scaffolding Cooperation")
    print("="*50)
    print(f"Device       : {TrainConfig.DEVICE}")
    print(f"Seed         : {args.seed}")
    print(f"Max Episodes : {TrainConfig.MAX_EPISODES}")
    print(f"Batch Size   : {TrainConfig.BATCH_SIZE}")
    print(f"Learning Rate: {TrainConfig.LR}")
    print("-" * 50)

    # 4. 初始化训练器
    try:
        trainer = Trainer()
    except Exception as e:
        print(f"\n[Error] Failed to initialize Trainer: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 开始训练
    start_time = time.time()
    print(f"Training started at {time.strftime('%X')}...")
    print("-" * 50)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n[Warning] Training interrupted by user (Ctrl+C).")
        # 这里可以选择保存紧急 Checkpoint，但 Trainer 已经有定期保存机制
    except Exception as e:
        print(f"\n[Error] An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 6. 结束统计
        end_time = time.time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        print("="*50)
        print(f"Training Process Finished.")
        print(f"Total Time: {hours}h {minutes}m {seconds}s")
        print(f"Checkpoints location: ./checkpoints/")
        print("="*50)

if __name__ == "__main__":
    main()