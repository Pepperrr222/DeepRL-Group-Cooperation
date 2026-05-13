# planners/graphnet.py
import torch
import os
from model.agent import SocialPlannerAgent
from .base import BasePlanner

class GraphNetPlanner(BasePlanner):
    def __init__(self, model_path, device):
        self.agent = SocialPlannerAgent().to(device)
        if os.path.exists(model_path):
            # map_location 确保可以在 CPU 上加载 GPU 训练的模型
            state_dict = torch.load(model_path, map_location=device)
            self.agent.load_state_dict(state_dict)
            
            # 设置为评估模式，关闭 Dropout 和 BatchNorm 等的训练特性
            self.agent.eval() 
            print(f"[GraphNet] 模型已加载: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

    def get_logits(self, capital, prev_decisions, edge_features, round_num):
        """
        获取 Agent 的动作 Logits。
        注意：在 V2 中，第三个参数是 edge_features (B, N, N, 2)
        """
        # 极其重要：推理时必须关闭梯度计算，否则会内存泄漏/显存爆炸
        with torch.no_grad():
            logits, _ = self.agent(capital, prev_decisions, edge_features, round_num)
            
        return logits