import requests
import json
import re

# ================= 配置区域 =================
# 请务必填入你正在使用的 Key
API_KEY = "sk-aonzxraxsctwtfshddtbaytnqpikuwssvhendbhhizohiaol" 

# 你的校内 API 地址
URL = "https://api.siliconflow.cn/v1/chat/completions"

# 模型名称 (建议先用 deepseek-chat 测试，因为它听话)
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct" 
# ===========================================

def test_decision_capability():
    print(f"🕵️‍♂️ 正在诊断 API 决策能力...")
    print(f"目标 URL: {URL}")
    print(f"使用模型: {MODEL_NAME}")
    print("-" * 50)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # 构造一个“极度诱导合作”的场景
    # 如果在这个场景下 LLM 都不回复 1，那说明有问题
    test_prompt = (
        "当前是第 1 回合。\n"
        "你有 5 个邻居，上一轮这 5 个人【全部】选择了合作。\n"
        "如果你选择合作，大家都能赚大钱。\n"
        "你会合作吗？\n"
        "**请仅回复数字 '1' (代表合作) 或 '0' (代表背叛)**，不要输出任何其他文字。"
    )

    data = {
        "messages": [
            {"role": "system", "content": "你是一个理性的博弈玩家。请严格遵循输出格式。"},
            {"role": "user", "content": test_prompt}
        ],
        "model": MODEL_NAME,
        "temperature": 0.1, # 极低温度，强迫它听话
        "stream": False
    }

    try:
        response = requests.post(URL, headers=headers, json=data, timeout=60)
        
        print(f"📡 HTTP 状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            # 1. 打印原始完整回复
            raw_content = result["choices"][0]["message"]["content"]
            print(f"\n📝 [重要] LLM 原始回复内容:\n{'-'*20}\n{raw_content}\n{'-'*20}")
            
            # 2. 模拟 llm_bots.py 里的清洗逻辑
            clean_content = raw_content.strip()
            if "</think>" in clean_content:
                clean_content = clean_content.split("</think>")[-1].strip()
            
            print(f"🧹 清洗后内容: [{clean_content}]")
            
            # 3. 模拟 llm_bots.py 里的提取逻辑
            digits = re.findall(r'\b[01]\b', clean_content)
            print(f"🔍 正则提取结果: {digits}")
            
            if digits:
                final_action = int(digits[0])
                print(f"✅ 最终判定动作: {final_action} ({'合作' if final_action==1 else '背叛'})")
            else:
                print("❌ 提取失败！代码将默认为 0 (背叛)。")
                print("   -> 原因可能是 LLM 回复了多余的标点或文字，导致正则不匹配。")
        else:
            print(f"❌ API 请求失败: {response.text}")

    except Exception as e:
        print(f"❌ 发生异常: {e}")

if __name__ == "__main__":
    test_decision_capability()