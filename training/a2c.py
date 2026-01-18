# training/a2c.py
import torch
import torch.nn.functional as F
from config import TrainConfig

def compute_a2c_loss(log_probs, values, rewards, entropies):
    """
    严格基于 Supp E2 描述计算 A2C 损失。
    
    参数:
    - log_probs: List of (B,) tensors. sum(log_pi_edge) for the graph.
    - values: List of (B, 1) tensors. Estimated state value omega(s).
    - rewards: List of (B,) tensors. U_SP at each step.
    - entropies: List of (B,) tensors. Mean entropy of the policy.
    """
    
    # --- 1. 计算回报 (Returns / Observed State Value) ---
    # Text: "sum of the rewards accumulated from that point forward"
    # 虽然 E2 文本公式未显式写出 Gamma，但 Table 5 定义了 Discount (gamma) = 0.99
    R = 0
    returns = []
    
    # 反向遍历计算累积回报 G_t
    for r in rewards[::-1]:
        R = r + TrainConfig.GAMMA * R
        returns.insert(0, R)
    
    returns = torch.stack(returns) # (T, B)
    values = torch.stack(values).squeeze(-1) # (T, B)
    log_probs = torch.stack(log_probs) # (T, B)
    entropies = torch.stack(entropies) # (T, B)
    
    # --- 2. 计算优势 (Advantage At) ---
    # Text: "difference between the value estimate ... and the sum of the rewards"
    # At = Gt - V(st)
    # Normalize advantage is a standard A2C practice for stability
    advantage = returns - values
    advantage_normalized = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    
    # --- 3. 计算梯度组件 (Gradient Components) ---
    
    # A. Policy Gradient
    # Text: "update w in the direction of ... grad log pi(st) * At"
    # 我们最小化 Loss，所以加负号
    # V-trace Note: 论文提到 "use a standard V-trace policy correction".
    # 在这个同步(Synchronous)实现中，Behavior Policy == Target Policy，
    # 因此 V-trace 裁剪系数 rho = 1，公式退化为标准 A2C。
    policy_loss = -(log_probs * advantage_normalized.detach()).mean()
    
    # B. Baseline Loss (Value Loss)
    # Text: "gradient of the l2 loss between the estimated and observed state value"
    # L2 Loss = (Gt - V(st))^2
    # Table 5 defines "Baseline cost" weight (usually 0.5)
    critic_loss = F.mse_loss(values, returns)
    
    # C. Entropy Regularization
    # Text: "regularize the policy with an entropy loss"
    # Table 5 defines "Entropy regularization" weight
    entropy_loss = -entropies.mean()
    
    # --- 4. 总损失 (Final Update) ---
    # Text: "We sum the three gradient components outlined above"
    total_loss = (policy_loss + 
                  TrainConfig.VALUE_LOSS_COEF * critic_loss + 
                  TrainConfig.ENTROPY_COEF * entropy_loss)
            
    return total_loss, policy_loss.item(), critic_loss.item(), entropy_loss.item()