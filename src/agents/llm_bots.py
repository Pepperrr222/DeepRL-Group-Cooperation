import numpy as np
import requests
import time
import random
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

class LLMBot:
    def __init__(self, num_players, api_key, base_url, model_name="deepseek-ai/DeepSeek-V3", mock=False):
        self.num_players = num_players
        self.api_key = api_key
        self.model_name = model_name
        self.mock = mock
        self.last_actions = np.zeros(num_players)
        
        # [修改点] 完全信任用户提供的完整 URL，不做任何自动拼接
        self.url = base_url
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        print(f"🤖 LLMBot 就绪 | 完整API地址: {self.url} | 模型: {self.model_name}")

        self.system_prompt = (
            "你正在参与一个'网络公共品博弈'游戏。你的目标是最大化自己的收益。\n"
            "规则：\n"
            "1. 选择'1' (合作) 成本 0.05，邻居各得 0.1。\n"
            "2. 选择'0' (背叛) 无成本，邻居无收益。\n"
            "3. 你的收益 = (邻居合作数 * 0.1) - (如果你合作 * 0.05 * 邻居总数)。\n"
            "请基于理性和互惠原则进行决策。"
        )

    def _call_api(self, user_prompt):
        """无阻塞的快速 API 调用"""
        if self.mock:
            if "合作" in user_prompt or "背叛" in user_prompt: return str(random.choice([0, 1]))
            if "Yes" in user_prompt or "No" in user_prompt: return random.choice(["Yes", "No"])
            return "0"

        data = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "model": self.model_name,
            "temperature": 0.3,
            "stream": False
        }

        # 重试 3 次
        for _ in range(3):
            try:
                # 只有微小的延迟
                time.sleep(0.1) 
                
                response = requests.post(self.url, headers=self.headers, json=data, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    if "</think>" in content: content = content.split("</think>")[-1].strip()
                    
                    # 快速提取
                    if "1" in content: return "1"
                    if "0" in content: return "0"
                    if "yes" in content.lower(): return "Yes"
                    if "no" in content.lower(): return "No"
                    return content #最后兜底
                
                elif response.status_code == 429:
                    print("⚠️ 触发429，稍等 2秒 重试...")
                    time.sleep(2)
                else:
                    print(f"⚠️ API Error {response.status_code}: {response.text}")
                    # 如果是404，通常意味着 URL 填错了，break 避免无效重试
                    if response.status_code == 404:
                        break
                    
            except Exception as e:
                print(f"⚠️ Net Error: {e}")
                
        return "0" # 默认背叛

    def decide_cooperation(self, adj_matrix, current_round):
        """[并行版] 16个线程并发请求"""
        actions = np.zeros(self.num_players)
        print(f"\n--- Round {current_round} 并发决策中... ---")
        
        def get_decision(i):
            neighbors = np.where(adj_matrix[i] == 1)[0]
            if len(neighbors) == 0: return i, 0
            
            coop_neighbors = sum(1 for n in neighbors if self.last_actions[n] == 1)
            
            if current_round == 0:
                prompt = (
                    f"这是游戏的第 1 回合（初始回合）。你有 {len(neighbors)} 个邻居。\n"
                    "没有历史记录。作为一个有远见的玩家，你愿意**率先释放善意，通过合作建立信任**吗？\n"
                    "**请仅回复数字 '1' (代表合作) 或 '0' (代表背叛)**。"
                )
            else:
                prompt = (
                    f"当前是第 {current_round + 1} 回合。你有 {len(neighbors)} 个邻居，"
                    f"上一轮有 {coop_neighbors} 人合作。\n"
                    "根据互惠原则，你会合作吗？\n"
                    "**请仅回复数字 '1' (代表合作) 或 '0' (代表背叛)**。"
                )
            
            reply = self._call_api(prompt)
            return i, 1 if "1" in reply else 0

        # 开启 16 个线程
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(get_decision, i) for i in range(self.num_players)]
            for future in concurrent.futures.as_completed(futures):
                i, action = future.result()
                actions[i] = action

        coop_count = int(sum(actions))
        print(f"⚡ 决策完毕。本轮合作人数: {coop_count}/{self.num_players}")
        self.last_actions = actions
        return actions

    def decide_acceptance(self, u, v, action_type, partner_last_action):
        """决定是否接受 Planner 的建议"""
        action_str = "建立连接" if action_type == 1 else "断开连接"
        partner_behavior = "合作" if partner_last_action == 1 else "背叛"
        
        prompt = (
            f"AI建议你与 Player {v} {action_str}。对方上一轮: {partner_behavior}。\n"
            "你接受吗？**仅回复 'Yes' 或 'No'**。"
        )
        reply = self._call_api(prompt)
        return "yes" in reply.lower()