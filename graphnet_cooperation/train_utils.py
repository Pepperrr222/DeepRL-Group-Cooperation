import torch
import torch.nn.functional as F


def discounted_returns(rewards, gamma: float, device=None):
    out = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32, device=device)


def a2c_episode_loss(log_probs, values, rewards, entropies, gamma=0.99, baseline_cost=0.5, entropy_coef=0.004):
    values = torch.stack(values)
    device = values.device

    returns = discounted_returns(rewards, gamma=gamma, device=device)
    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)

    advantages = returns - values.detach()

    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_bonus = entropies.mean()

    total_loss = policy_loss + baseline_cost * value_loss - entropy_coef * entropy_bonus

    return total_loss, {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy_bonus.item()),
        "mean_return": float(returns.mean().item()),
    }