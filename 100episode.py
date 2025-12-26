import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.training.trainer import SocialPlannerTrainer


def run_n_episodes(n=100, num_players=16, max_rounds=15):
    trainer = SocialPlannerTrainer(num_players=num_players)
    trainer.planner.eval()

    final_coop = []
    for i in range(n):
        hist = trainer.run_episode_record(max_rounds=max_rounds)
        last_actions = np.array(hist['actions'][-1])
        final_coop.append(last_actions.mean())
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{n} episodes")

    return np.array(final_coop)


def plot_results(final_coop, out_path='outputs/cooperation_100.png'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(final_coop, marker='o', linestyle='-', alpha=0.7)
    plt.xlabel('Run')
    plt.ylabel('Final-round cooperation rate')
    plt.ylim(-0.05, 1.05)
    plt.title(f'Final cooperation over {len(final_coop)} runs (per-run last round mean)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    # also save histogram
    hist_path = out_path.replace('.png', '_hist.png')
    plt.figure(figsize=(6, 4))
    plt.hist(final_coop, bins=10, range=(0, 1), color='skyblue', edgecolor='k')
    plt.xlabel('Final cooperation rate')
    plt.ylabel('Count')
    plt.title('Distribution of final cooperation')
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=int, default=100)
    p.add_argument('--players', type=int, default=16)
    p.add_argument('--rounds', type=int, default=15)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    final_coop = run_n_episodes(n=args.runs, num_players=args.players, max_rounds=args.rounds)
    print(f"Mean final cooperation over {len(final_coop)} runs: {final_coop.mean():.3f}")
    plot_results(final_coop, out_path=os.path.join('outputs', f'cooperation_{args.runs}.png'))
    print('Saved plots to outputs/')

