# env/llm_bots.py
import torch
import re
import concurrent.futures
from config import GameConfig
# 假设你使用 openai 库，也可以换成 requests 调用其它 API
from openai import OpenAI 

class LLMBots:
    def __init__(self, batch_size, device, api_key, base_url=None, model="gpt-4o-mini"):
        """
        注意：LLM 评估时，强烈建议 batch_size 只能设为 1，否则 API 并发量太大
        """
        assert batch_size == 1, "LLM bots only support batch_size=1 for evaluation."
        self.bs = batch_size
        self.device = device
        self.n_players = GameConfig.N_PLAYERS
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        # 给每个玩家分配一个固定的人设（可选），或者统一设定为自利者
        self.system_prompt = (
            "You are a human participant in a public goods game played on a social network. "
            "Your goal is to maximize your own virtual money (capital). "
            "Think logically based on the actions of your neighbors."
        )

    def _call_llm(self, prompt):
        """调用 LLM API，获取文本回答"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3, # 稍微降低随机性，保持逻辑一致
                max_tokens=10 # 只需要回复单个词
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"API Error: {e}")
            return ""

    def decide_cooperation(self, round_num, adj_matrix, prev_decisions, current_capital):
        # 降维取出 batch=0 的数据
        adj = adj_matrix[0] 
        prev = prev_decisions[0]
        cap = current_capital[0]
        
        degrees = adj.sum(dim=1)
        coop_neighbors = (adj * prev).sum(dim=1)
        
        prompts =[]
        for i in range(self.n_players):
            deg = int(degrees[i].item())
            cost = GameConfig.COST_C * deg
            my_cap = cap[i].item()
            
            # 破产保护：如果钱不够，LLM 连选的资格都没有，直接跳过生成 Prompt
            if my_cap < cost:
                prompts.append(None)
                continue
                
            coop_n = int(coop_neighbors[i].item())
            
            prompt = (
                f"Round {round_num + 1}.\n"
                f"You currently have ${my_cap:.2f}.\n"
                f"You have {deg} connected neighbors. Last round, {coop_n} of them chose to Cooperate.\n"
                f"If you Cooperate, you pay ${cost:.2f}, and each neighbor gets ${GameConfig.BENEFIT_B:.2f}.\n"
                f"If you Defect, you pay $0, but you still receive money if neighbors cooperate.\n"
                f"Will you 'Cooperate' or 'Defect'? Reply ONLY with one word."
            )
            prompts.append(prompt)

        # 使用多线程并行请求 API，否则 16 个人串行会等很久
        decisions = torch.zeros(self.n_players, device=self.device)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # 提交任务
            future_to_player = {
                executor.submit(self._call_llm, p): i 
                for i, p in enumerate(prompts) if p is not None
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_player):
                i = future_to_player[future]
                reply = future.result()
                if "cooperate" in reply:
                    decisions[i] = 1.0
                elif "defect" in reply:
                    decisions[i] = 0.0
                else:
                    # 如果 LLM 胡言乱语，默认背叛
                    decisions[i] = 0.0

        return decisions.unsqueeze(0) # 重新加上 batch 维度 (1, N)

    def decide_acceptance(self, recommendations, prev_decisions):
        rec = recommendations[0]
        prev = prev_decisions[0]
        
        accept_mask = torch.zeros_like(rec)
        prompts_info =[]

        # 遍历上三角，寻找被推荐的边
        for i in range(self.n_players):
            for j in range(i + 1, self.n_players):
                if rec[i, j] != 0:
                    action_type = "CONNECT TO" if rec[i, j] == 1 else "DISCONNECT FROM"
                    # 这里为了简化，我们让节点 i 做决定，或者询问双方。
                    # 论文中是：assigned randomly to one of the two players。这里固定指派给 i 决定。
                    partner_action = "Cooperated" if prev[j] == 1 else "Defected"
                    
                    prompt = (
                        f"The Social Planner recommends you to {action_type} Player {j}.\n"
                        f"Last round, Player {j} {partner_action}.\n"
                        f"Will you 'Accept' or 'Reject' this recommendation? Reply ONLY with one word."
                    )
                    prompts_info.append((i, j, prompt))

        # 并发请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_edge = {
                executor.submit(self._call_llm, p): (i, j) 
                for i, j, p in prompts_info
            }
            
            for future in concurrent.futures.as_completed(future_to_edge):
                i, j = future_to_edge[future]
                reply = future.result()
                if "accept" in reply:
                    accept_mask[i, j] = 1.0
                    accept_mask[j, i] = 1.0 # 保持对称

        return accept_mask.unsqueeze(0)