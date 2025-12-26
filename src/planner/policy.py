import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

class SocialPlannerPolicy:
    """
    策略模块：负责将神经网络的 Logits 转换为具体的动作 (Action)
    """
    def __init__(self):
        pass

    def get_action(self, edge_logits, deterministic=False):
        """
        根据 logits 采样动作
        
        参数:
            edge_logits: [batch_size, N, N] 模型输出的原始数值
            deterministic: bool, 如果为True，则直接选概率最大的动作(用于测试/演示)；
                           如果为False，则按照概率分布采样(用于训练)。
        
        返回:
            actions: [batch_size, N, N] 0或1的矩阵，表示Planner建议的连边结构
            log_probs: [batch_size] 此次采样的对数概率 (用于RL Loss计算)
            entropy: [batch_size] 熵 (衡量策略的随机程度，常用于鼓励探索)
        """
        # 1. 将 Logits 转换为概率 (Sigmoid: (-inf, inf) -> (0, 1))
        # 比如 logit=0 -> prob=0.5; logit=2 -> prob=0.88
        probs = torch.sigmoid(edge_logits)
        
        # 2. 构建伯努利分布 (Bernoulli Distribution)
        # 这就像是给每一条边都准备了一枚不均匀的硬币
        dist = Bernoulli(probs=probs)
        
        # 3. 采样动作
        if deterministic:
            # 测试时：概率>0.5就连，否则断
            actions = (probs > 0.5).float()
        else:
            # 训练时：掷硬币决定
            actions = dist.sample()
            
        # 4. 计算 Log Probability
        # 我们需要知道“整张图”被采样出来的概率是多少
        # sum(-1).sum(-1) 表示把所有边的 log_prob 加起来
        log_probs = dist.log_prob(actions).sum(dim=(-1, -2))
        
        # 5. 计算熵 (Entropy)
        # 熵越高，表示分布越随机（探索性越强）
        entropy = dist.entropy().sum(dim=(-1, -2))
        
        return actions, log_probs, entropy

# --- 简单测试 ---
if __name__ == "__main__":
    # 模拟一个 2个batch, 4个玩家 的 logits
    # 假设模型输出了一些随机数
    fake_logits = torch.randn(2, 4, 4)
    
    policy = SocialPlannerPolicy()
    
    # 尝试采样
    actions, log_probs, entropy = policy.get_action(fake_logits)
    
    print("Logits (部分):", fake_logits[0, 0])
    print("Actions (部分):", actions[0, 0])
    print(f"Log Probs shape: {log_probs.shape} (应为 [2])")
    print(f"Values: {log_probs}")