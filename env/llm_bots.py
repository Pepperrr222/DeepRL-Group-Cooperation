# env/llm_bots.py
import torch
import re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from config import BotConfig, GameConfig

class LLMBots:
    def __init__(self, batch_size, device, model_name="gpt-3.5-turbo"):
        self.bs = batch_size
        self.device = device
        self.n_players = GameConfig.N_PLAYERS
        self.model_name = model_name
        
        # 初始化客户端 (请确保已设置环境变量 OPENAI_API_KEY)
        # 如果使用其他模型，可以在这里修改 base_url
        self.client = OpenAI() 
        
        # 依然保留个体性格 theta，作为 Prompt 的输入之一
        self.theta = torch.normal(
            BotConfig.MU_THETA, 
            BotConfig.SIGMA_THETA, 
            size=(self.bs, self.n_players), 
            device=self.device
        )

    def _call_llm(self, prompt, fallback=0):
        """调用 LLM 并解析返回的 0 或 1"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a participant in a sociological game. You must reply with exactly one digit: 1 or 0. No other text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3, # 低温度保证输出稳定
                max_tokens=5
            )
            reply = response.choices[0].message.content.strip()
            
            # 使用正则提取第一个出现的 0 或 1
            match = re.search(r'[01]', reply)
            if match:
                return float(match.group(0))
            return float(fallback)
        except Exception as e:
            print(f"[LLM Error] API request failed: {e}. Using fallback {fallback}.")
            return float(fallback)

    def decide_cooperation(self, round_num, adj_matrix, prev_decisions, current_capital):
        """1. 合作决策"""
        x_s = adj_matrix.sum(dim=2) # 邻居总数
        prev_decisions_exp = prev_decisions.unsqueeze(1).expand(-1, self.n_players, -1)
        x_n = (adj_matrix * prev_decisions_exp).sum(dim=2) # 合作的邻居数
        
        x_r = torch.zeros_like(x_s)
        mask_degree = x_s > 0
        x_r[mask_degree] = x_n[mask_degree] / x_s[mask_degree] # 合作率

        initial_decisions = torch.zeros((self.bs, self.n_players), device=self.device)
        prompts =[]

        # 构造 Prompts
        for b in range(self.bs):
            for i in range(self.n_players):
                t = self.theta[b, i].item()
                total_neighbors = int(x_s[b, i].item())
                coop_neighbors = int(x_n[b, i].item())
                coop_rate = x_r[b, i].item()
                
                if round_num == 0:
                    prompt = (f"This is round 1. You have no history yet. "
                              f"Your personality score is {t:.2f} (higher means more cooperative). "
                              f"Do you choose to cooperate (1) or defect (0)?")
                else:
                    prompt = (f"Your personality score is {t:.2f} (higher means more cooperative). "
                              f"Last round, you had {total_neighbors} connected neighbors, "
                              f"and {coop_neighbors} of them cooperated (cooperation rate: {coop_rate:.0%}). "
                              f"Based on this, do you choose to cooperate (1) or defect (0) this round?")
                prompts.append(prompt)

        # 多线程并发请求 LLM
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(lambda p: self._call_llm(p, fallback=0), prompts))

        # 将结果填入 Tensor
        idx = 0
        for b in range(self.bs):
            for i in range(self.n_players):
                initial_decisions[b, i] = results[idx]
                idx += 1

        # 依然保留资金不足强制背叛的物理规则 (Bankruptcy Protection)
        potential_cost = GameConfig.COST_C * x_s
        cannot_afford_mask = current_capital < potential_cost
        
        final_decisions = initial_decisions.clone()
        final_decisions[cannot_afford_mask] = 0.0
        
        return final_decisions

    def decide_acceptance(self, recommendations, prev_decisions):
        """2. 接受/拒绝边建议决策"""
        B, N, _ = recommendations.shape
        accept_decisions = torch.zeros_like(recommendations, dtype=torch.float)
        
        partner_prev = prev_decisions.unsqueeze(1).expand(-1, N, -1)
        
        prompts = []
        indices =[] # 记录需要请求LLM的索引位置 (b, i, j)

        # 遍历寻找所有 Agent 提出的建议
        for b in range(B):
            for i in range(N):
                for j in range(i+1, N): # 无向图，只看上三角
                    rec = recommendations[b, i, j].item()
                    if rec != 0: # Agent 有建议 (-1 或 1)
                        action_str = "create a new connection with" if rec == 1 else "cut the existing connection with"
                        partner_action = "cooperated" if partner_prev[b, i, j].item() == 1 else "defected"
                        
                        prompt = (f"The social planner recommends that you {action_str} Player {j}. "
                                  f"Last round, Player {j} {partner_action}. "
                                  f"Do you accept (1) or reject (0) this recommendation?")
                        
                        prompts.append(prompt)
                        indices.append((b, i, j))

        # 并发请求
        if prompts:
            with ThreadPoolExecutor(max_workers=16) as executor:
                results = list(executor.map(lambda p: self._call_llm(p, fallback=1), prompts))
                
            # 填回 Tensor (并且保持对称性)
            for k, (b, i, j) in enumerate(indices):
                ans = results[k]
                accept_decisions[b, i, j] = ans
                accept_decisions[b, j, i] = ans # 双方同时决定，简化为一侧决定即代表该边

        return accept_decisions