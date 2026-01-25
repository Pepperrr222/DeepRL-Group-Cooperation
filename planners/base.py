import torch

def prob_to_logits(p_change, device):
    """
    工具函数：将改变的概率 p 转换为 logits [Keep, Change]
    """
    # 限制 p 防止 log(0)
    if isinstance(p_change, torch.Tensor):
        p = torch.clamp(p_change, 0.001, 0.999)
        val_keep = torch.log(1 - p)
        val_change = torch.log(p)
        return torch.stack([val_keep, val_change], dim=-1)
    else:
        
        # 处理标量情况
        p = max(0.001, min(0.999, p_change))
        import math
        return torch.tensor([math.log(1-p), math.log(p)], device=device)

class BasePlanner:
    def get_logits(self, capital, prev_decisions, adj, round_num):
        """
        所有 Planner 必须实现此方法
        返回: Logits (B, N, N, 2)
        """
        raise NotImplementedError