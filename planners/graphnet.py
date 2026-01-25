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
            self.agent.eval()
            print(f"[GraphNet] 模型已加载: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

    def get_logits(self, capital, prev_decisions, adj, round_num):
        # 调用模型的 forward
        logits, _ = self.agent(capital, prev_decisions, adj, round_num)
        return logits