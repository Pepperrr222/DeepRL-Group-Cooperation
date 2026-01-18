# utils/init_weights.py
import torch
import torch.nn as nn
import math

def truncated_normal_(tensor, mean=0, std=1):
    """
    实现论文中提到的截断正态分布初始化。
    截断范围通常为 [-2std, 2std]。
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Supp E5: sigma = 1 / sqrt(input_size)
        input_dim = m.weight.size(1)
        std = 1.0 / math.sqrt(input_dim)
        truncated_normal_(m.weight, mean=0, std=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)