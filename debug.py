import numpy as np
import requests
import re
from concurrent.futures import ThreadPoolExecutor

# ================= 配置 =================
API_KEY = "sk-aonzxraxsctwtfshddtbaytnqpikuwssvhendbhhizohiaol"
BASE_URL = "https://api.siliconflow.cn/v1/chat/completions" # 你的 API 地址
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # 或 deepseek-ai/DeepSeek-V3
# =======================================

class DebugBot:
    def __init__(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        self.url = BASE_URL
        if "/chat/completions" not in self.url:
             self.url += "/chat/completions"

    def _call_api(self, prompt):
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": MODEL_NAME,
            "temperature": 0.1 # 低温，便于复现
        }
        try:
            resp = requests.post(self.url, headers=self.headers, json=data, timeout=30)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            return f"Error {resp.status_code}"
        except Exception as e:
            return f"Exception {e}"

    def robust_parse_acceptance(self, reply):
        """
        更强健的解析逻辑 (这是我们打算替换旧代码的逻辑)
        """
        text = reply.lower()
        # 关键词列表：只要出现其中一个，就认为是同意
        positive_keywords = ["yes", "accept", "agree", "sure", "ok", "willing", "同意", "接受"]
        
        for word in positive_keywords:
            # 使用正则匹配单词边界，防止把 "yesterday" 识别为 "yes"
            # 但为了简单，先直接看包含
            if word in text:
                return True, word # 返回结果和匹配到的词
        return False, None

def run_debug():
    bot = DebugBot()
    
    print(f"🕵️‍♂️ 开始诊断解析逻辑 (模型: {MODEL_NAME})...\n")

    # === 测试 1: 接受建议 (Acceptance) ===
    print("Test 1: 模拟 Planner 建议连接合作者")
    prompt_accept = (
        "AI建议你与 Player 5 建立连接。对方上一轮选择了: 合作。\n"
        "连接合作者对你有益。\n"
        "你接受吗？请回答。" # 故意不给格式提示，看它怎么回
    )
    
    reply = bot._call_api(prompt_accept)
    print(f" -> 🤖 原始回复: [{reply}]")
    
    # 用旧逻辑判断
    old_result = "yes" in reply.lower()
    print(f" -> ❌ 旧代码判定: {old_result} (只找 'yes')")
    
    # 用新逻辑判断
    new_result, keyword = bot.robust_parse_acceptance(reply)
    print(f" -> ✅ 新代码判定: {new_result} (匹配词: {keyword})")
    print("-" * 50)

    # === 测试 2: 合作决策 (Cooperation) ===
    print("\nTest 2: 模拟决策 (诱导合作)")
    prompt_coop = (
        "你的性格是：利他主义者。\n"
        "你有 5 个邻居，上一轮全部合作。\n"
        "你会合作吗？请回答。"
    )
    
    reply = bot._call_api(prompt_coop)
    print(f" -> 🤖 原始回复: [{reply}]")
    
    # 提取数字
    digits = re.findall(r'\b[01]\b', reply)
    print(f" -> 🔢 数字提取: {digits}")
    
    # 提取文字意图
    has_coop = "合作" in reply or "cooperate" in reply.lower()
    print(f" -> 🔤 文字提取: {'合作' if has_coop else '未检测到'}")

if __name__ == "__main__":
    run_debug()