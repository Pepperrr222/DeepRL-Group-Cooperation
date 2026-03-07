from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import numpy as np
import torch
from torch.distributions import Categorical

from graphnet_cooperation.env import CooperationGameEnv, GameConfig
from graphnet_cooperation.model import GraphNetPlanner
from graphnet_cooperation.train_utils import a2c_episode_loss


def to_tensors(state, device):
    node_features = torch.tensor(state["node_features"], dtype=torch.float32, device=device)
    edge_features = torch.tensor(state["edge_features"], dtype=torch.float32, device=device)
    global_features = torch.tensor(state["global_features"], dtype=torch.float32, device=device)
    edge_pairs = state["edge_pairs"]
    n_nodes = state["node_features"].shape[0]
    return node_features, edge_features, global_features, edge_pairs, n_nodes


def run_episode(env, model, device):
    state = env.reset()

    episode_log_probs = []
    episode_values = []
    episode_rewards = []
    episode_entropies = []

    coop_hist = []
    cap_hist = []

    done = False
    while not done:
        node_features, edge_features, global_features, edge_pairs, n_nodes = to_tensors(state, device)

        edge_logits, value = model(
            node_features=node_features,
            edge_features=edge_features,
            global_features=global_features,
            edge_pairs=edge_pairs,
            n_nodes=n_nodes,
        )

        dist = Categorical(logits=edge_logits)
        sampled = dist.sample()

        planner_actions = {}
        for k, pair in enumerate(edge_pairs):
            planner_actions[pair] = int(sampled[k].item())

        mean_log_prob = dist.log_prob(sampled).mean()
        mean_entropy = dist.entropy().mean()

        next_state, reward, done, info = env.step(planner_actions)

        episode_log_probs.append(mean_log_prob)
        episode_values.append(value)
        episode_rewards.append(float(reward))
        episode_entropies.append(mean_entropy)

        coop_hist.append(info["mean_cooperation"])
        cap_hist.append(info["mean_capital"])

        state = next_state

    summary = {
        "Mean_Game_Cooperation": float(np.mean(coop_hist)),
        "Mean_Capital": float(np.mean(cap_hist)),
    }

    return episode_log_probs, episode_values, episode_rewards, episode_entropies, summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Path("results").mkdir(exist_ok=True)
    Path("figs").mkdir(exist_ok=True)

    cfg = GameConfig()
    env_seed_base = 1000

    model = GraphNetPlanner(node_dim=3, edge_dim=1, global_dim=2, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

    gamma = 0.99
    entropy_coef = 0.004
    baseline_cost = 0.5

    # 轻量版：只跑 500 局
    n_episodes = 500

    log_path = Path("results/our_training_log.csv")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Mean_Game_Cooperation", "Mean_Capital", "Loss"])

        for episode in range(1, n_episodes + 1):
            env = CooperationGameEnv(cfg, seed=env_seed_base + episode)

            log_probs, values, rewards, entropies, summary = run_episode(env, model, device)

            loss, loss_info = a2c_episode_loss(
                log_probs=log_probs,
                values=values,
                rewards=rewards,
                entropies=entropies,
                gamma=gamma,
                baseline_cost=baseline_cost,
                entropy_coef=entropy_coef,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            writer.writerow([
                episode,
                summary["Mean_Game_Cooperation"],
                summary["Mean_Capital"],
                float(loss.item()),
            ])

            if episode % 50 == 0:
                print(
                    f"Episode {episode:04d} | "
                    f"Mean Coop {summary['Mean_Game_Cooperation']:.3f} | "
                    f"Mean Capital {summary['Mean_Capital']:.3f} | "
                    f"Loss {loss.item():.4f}"
                )
                torch.save(model.state_dict(), f"results/checkpoint_episode_{episode:04d}.pt")

    torch.save(model.state_dict(), "results/graphnet_a2c_first_pass.pt")
    print("Training finished.")
    print("Saved:")
    print(" - results/our_training_log.csv")
    print(" - results/graphnet_a2c_first_pass.pt")


if __name__ == "__main__":
    main()