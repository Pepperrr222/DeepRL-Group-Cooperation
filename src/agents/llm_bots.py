import numpy as np
import requests
import time
import random
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import re

class LLMBot:
    def __init__(self, num_players, api_key, base_url, model_name="deepseek-ai/DeepSeek-V3", mock=False):
        self.num_players = num_players
        self.api_key = api_key
        self.model_name = model_name
        self.mock = mock
        self.last_actions = np.zeros(num_players)
        
        # URL 处理
        self.url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 性格分配
        persona_types = [
            "一个乐于助人的'利他主义者'，你相信人性本善，愿意主动承担风险。",
            "一个精明的'利己主义者'，你极度厌恶损失，绝不当冤大头。",
            "一个冷静的'理性分析师'，你只看数据，如果大家都合作你就合作，否则就止损。",
            "一个谨慎的'怀疑论者'，在看到确凿的证据之前，你倾向于保持观望。"
        ]
        self.personas = [random.choice(persona_types) for _ in range(num_players)]
        
        print(f"🤖 LLMBot 就绪 | 解析逻辑: 增强版 (语义匹配)")

        self.system_prompt = (
            "你正在参与一个'网络公共品博弈'游戏。你的目标是最大化自己的收益。\n"
            "规则：合作(1)成本0.05，邻居得0.1；背叛(0)无成本。\n"
            "请基于你的性格设定进行决策。"
        )

    def _call_api(self, user_prompt):
        """发送请求并返回清洗后的文本 (不负责具体解析)"""
        if self.mock:
            if "合作" in user_prompt: return "我选择合作" if random.random()>0.5 else "背叛"
            return "接受建议" if random.random()>0.5 else "拒绝"

        data = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "model": self.model_name,
            "temperature": 0.4,
            "stream": False
        }

        for _ in range(3):
            try:
                time.sleep(0.1)
                response = requests.post(self.url, headers=self.headers, json=data, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    # 清洗思维链
                    if "</think>" in content: content = content.split("</think>")[-1].strip()
                    return content
                
                elif response.status_code == 429:
                    time.sleep(2)
                elif response.status_code == 404:
                    print("⚠️ 404 Error: 请检查模型名称是否正确")
                    break
                else:
                    print(f"⚠️ API Error {response.status_code}")

            except Exception as e:
                print(f"⚠️ Net Error: {e}")
                
        return "0" # 兜底

    def decide_cooperation(self, adj_matrix, current_round):
        actions = np.zeros(self.num_players)
        print(f"\n--- Round {current_round} 并发决策中 (增强解析) ---")
        
        def get_decision(i):
            neighbors = np.where(adj_matrix[i] == 1)[0]
            if len(neighbors) == 0: return i, 0
            
            coop_neighbors = sum(1 for n in neighbors if self.last_actions[n] == 1)
            my_persona = self.personas[i]
            
            if current_round == 0:
                prompt = (
                    f"你的性格设定是：**{my_persona}**\n\n"
                    f"这是第 1 回合。你有 {len(neighbors)} 个邻居。没有历史记录。\n"
                    "如果你选择合作，你需要付出成本，且如果邻居背叛，你会亏损。\n"
                    "如果你选择背叛，你没有任何风险，但也可能失去建立长期合作的机会。\n"
                    "基于你的性格设定，你会怎么选？\n"
                    "**请回答：我选择合作(1) 还是 我选择背叛(0)**。"
                )
            else:
                prompt = (
                    f"你的性格设定是：**{my_persona}**\n\n"
                    f"当前是第 {current_round + 1} 回合。你有 {len(neighbors)} 个邻居，"
                    f"上一轮有 {coop_neighbors} 人合作。\n"
                    "基于你的性格和当前的局势，你会合作吗？\n"
                    "**请回答：我选择合作(1) 还是 我选择背叛(0)**。"
                )
            
            reply = self._call_api(prompt)
            
            # === [核心修复] 强力解析逻辑 ===
            # 1. 优先找数字
            if "1" in reply: return i, 1
            if "0" in reply: return i, 0
            
            # 2. 其次找关键词 (防止话痨)
            lower_reply = reply.lower()
            if "合作" in reply or "cooperate" in lower_reply: return i, 1
            if "背叛" in reply or "defect" in lower_reply: return i, 0
            
            # 3. 实在看不懂，保守背叛
            # print(f"  [解析失败] P{i} 回复: {reply[:20]}...") 
            return i, 0

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(get_decision, i) for i in range(self.num_players)]
            for future in concurrent.futures.as_completed(futures):
                i, action = future.result()
                actions[i] = action

        print(f"⚡ 决策完毕。本轮合作人数: {int(sum(actions))}/{self.num_players}")
        self.last_actions = actions
        return actions

    def decide_acceptance(self, u, v, action_type, partner_last_action):
        action_str = "建立连接" if action_type == 1 else "断开连接"
        partner_behavior = "合作" if partner_last_action == 1 else "背叛"
        
        prompt = (
            f"AI建议你与 Player {v} {action_str}。对方上一轮: {partner_behavior}。\n"
            "连接合作者有益，连接背叛者有害。\n"
            "你接受这个建议吗？**请回答 Yes 或 No**。"
        )
        reply = self._call_api(prompt)
        
        # === [核心修复] 强力解析逻辑 ===
        cleaned = reply.lower()
        
        # 同意词库
        positive_keywords = ["yes", "accept", "agree", "sure", "ok", "1", "接受", "同意", "好", "可以"]
        
        for word in positive_keywords:
            if word in cleaned:
                return True
                
        return False