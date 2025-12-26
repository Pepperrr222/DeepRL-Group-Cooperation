import time
import numpy as np
import torch

from src.training.trainer import SocialPlannerTrainer


def main(episodes=20):
    # 固定随机种子以便复现
    np.random.seed(0)
    torch.manual_seed(0)

    trainer = SocialPlannerTrainer(num_players=16,
                                   lr=0.0004,
                                   gamma=0.99,
                                   entropy_coef=0.004,
                                   batch_size=32)

    print(f"Start short training for {episodes} episodes on device {trainer.device}")

    history = {"coop": [], "rew": [], "loss": []}

    start = time.time()
    for ep in range(1, episodes + 1):
        metrics = trainer.run_episode(train=True)
        history['coop'].append(metrics['mean_cooperation'])
        history['rew'].append(metrics['total_reward'])
        history['loss'].append(metrics['loss'])

        print(f"Ep {ep}/{episodes} | Coop: {metrics['mean_cooperation']:.2%} | TotalReward: {metrics['total_reward']:.4f} | Loss: {metrics['loss']:.4f}")

    elapsed = time.time() - start
    print('-' * 40)
    print(f"Finished {episodes} episodes in {elapsed:.1f}s")
    print(f"Avg Coop: {np.mean(history['coop']):.2%} | Avg Reward: {np.mean(history['rew']):.4f} | Avg Loss: {np.mean(history['loss']):.4f}")


if __name__ == '__main__':
    main(episodes=20)
