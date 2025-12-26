import argparse
import torch
import numpy as np

from src.training.trainer import SocialPlannerTrainer


def evaluate(model_path, num_episodes=10, num_players=16):
    # 初始化训练器 (它封装了 env、bots、planner 和 policy 的接口)
    trainer = SocialPlannerTrainer(num_players=num_players)

    device = trainer.device

    # 加载模型参数
    state = torch.load(model_path, map_location=device)
    try:
        trainer.planner.load_state_dict(state)
    except Exception:
        # 如果保存的是包含额外键的 dict（比如 trainer state），尝试提取 planner
        if isinstance(state, dict) and 'planner' in state:
            trainer.planner.load_state_dict(state['planner'])
        else:
            raise

    trainer.planner.to(device)
    trainer.planner.eval()

    coop_rates = []
    total_rewards = []

    for ep in range(num_episodes):
        metrics = trainer.run_episode(train=False)
        coop_rates.append(metrics['mean_cooperation'])
        total_rewards.append(metrics['total_reward'])
        print(f"Episode {ep+1}/{num_episodes} | Coop: {metrics['mean_cooperation']:.2%} | TotalReward: {metrics['total_reward']:.4f}")

    print("-" * 40)
    print(f"Avg Cooperation Rate over {num_episodes} episodes: {np.mean(coop_rates):.2%}")
    print(f"Avg Total Reward over {num_episodes} episodes: {np.mean(total_rewards):.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='saved_models/social_planner_final.pth', help='Path to the saved model (.pth)')
    p.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    p.add_argument('--players', type=int, default=16, help='Number of players / nodes')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args.model, num_episodes=args.episodes, num_players=args.players)
