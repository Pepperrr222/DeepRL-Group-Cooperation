import argparse
import torch
import numpy as np
import sys

from src.training.trainer import SocialPlannerTrainer


def load_state_dict_robust(model, state_dict):
    """Try to load state_dict fully; if shapes mismatch, load matching keys only."""
    model_dict = model.state_dict()
    # direct load if keys identical
    try:
        model.load_state_dict(state_dict)
        print("Loaded state_dict exactly.")
        return
    except Exception as e:
        print("Exact load failed:", e)

    # if state is a dict with nested 'planner'
    if isinstance(state_dict, dict) and 'planner' in state_dict:
        candidate = state_dict['planner']
    else:
        candidate = state_dict

    # Partial load: only keep keys that exist in model and match shape
    to_load = {}
    skipped = []
    for k, v in candidate.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                to_load[k] = v
            else:
                skipped.append((k, v.shape, model_dict[k].shape))
        else:
            skipped.append((k, v.shape, None))

    if len(to_load) == 0:
        raise RuntimeError("No compatible parameters found in checkpoint for partial load.")

    model_dict.update(to_load)
    model.load_state_dict(model_dict)
    print(f"Partially loaded {len(to_load)} parameters; skipped {len(skipped)} entries.")
    if skipped:
        print("Some skipped keys (key, checkpoint_shape, model_shape):")
        for s in skipped[:10]:
            print(" ", s)


def evaluate(model_path, num_episodes=100, num_players=16, print_initial=True):
    trainer = SocialPlannerTrainer(num_players=num_players)
    device = trainer.device

    # optionally show initial cooperation before loading model
    if print_initial:
        probs = trainer.initial_cooperation(sample=False, seed=0)
        acts, _ = trainer.initial_cooperation(sample=True, seed=0)
        print(f"Initial mean cooperation probability (seed=0): {probs.mean():.3f}")
        print(f"Initial sampled cooperation rate (seed=0): {acts.mean():.3f}")

    # load checkpoint
    print(f"Loading checkpoint: {model_path}")
    state = torch.load(model_path, map_location=device)

    # Try to robustly load into planner
    try:
        load_state_dict_robust(trainer.planner, state)
    except Exception as e:
        print("Failed to load planner state:", e)
        print("Proceeding with current (randomly initialized) model. To use checkpoint, provide a matching checkpoint or retrain.")

    trainer.planner.to(device)
    trainer.planner.eval()

    coop_rates = []
    total_rewards = []

    for ep in range(num_episodes):
        # Use run_episode_record to capture per-round adjacency and actions
        history = trainer.run_episode_record(max_rounds=15)
        # compute metrics from history
        coop_rates.append(np.mean([actions.mean() for actions in history['actions'][1:]]))
        total_rewards.append(0.0)  # not used here
        print(f"Episode {ep+1}/{num_episodes} | Coop (avg over rounds): {coop_rates[-1]:.2%}")

        # Save visualizations for rounds 0,5,10,15 (if present)
        import os
        import matplotlib.pyplot as plt
        import networkx as nx

        out_dir = os.path.join('outputs', f'episode_{ep+1}')
        os.makedirs(out_dir, exist_ok=True)

        # Save plots for every recorded round
        num_rounds = len(history['adjs'])
        for rr in range(num_rounds):
            adj = history['adjs'][rr]
            actions = history['actions'][rr]

            # Binarize adjacency for plotting (threshold at 0.5)
            adj_bin = (np.array(adj) > 0.5).astype(int)

            G = nx.from_numpy_array(adj_bin)
            color_map = ['skyblue' if a == 1 else 'lightcoral' for a in actions]

            plt.figure(figsize=(6, 6))
            nx.draw(G, node_color=color_map, with_labels=True, node_size=500, font_weight='bold')
            plt.title(f'Episode {ep+1} - Round {rr}')
            save_path = os.path.join(out_dir, f'round_{rr}.png')
            plt.savefig(save_path)
            plt.close()

    print("-" * 40)
    print(f"Avg Cooperation Rate over {num_episodes} episodes: {np.mean(coop_rates):.2%}")
    print(f"Avg Total Reward over {num_episodes} episodes: {np.mean(total_rewards):.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='saved_models/social_planner_final.pth')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--players', type=int, default=16)
    p.add_argument('--no-initial', dest='print_initial', action='store_false')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args.model, num_episodes=args.episodes, num_players=args.players, print_initial=args.print_initial)

