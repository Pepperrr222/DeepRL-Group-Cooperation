# LLMBots 脚本详解

文件路径：env/llm_bots.py

---

## 一、导入部分

```python
import torch
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from config import BotConfig, GameConfig, LLMConfig
```

各模块用途：

- torch — 张量运算，所有决策结果以 tensor 形式返回
- re — 正则表达式，从 LLM 返回的文本中提取 0 或 1
- time — 限流等待和重试退避
- threading — 限流器的线程锁，多线程并发时防止竞态
- ThreadPoolExecutor — 并发调用多个 LLM 请求
- OpenAI — 兼容 OpenAI / DeepSeek / 通义等 API 的客户端
- BotConfig — 玩家性格参数（theta 的均值和标准差）
- GameConfig — 游戏配置（玩家数、轮次数等）
- LLMConfig — LLM 调用配置（API key、模型、温度、重试策略等）

---

## 二、RateLimiter 类

```python
class RateLimiter:
    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.last_request = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            wait_time = self.last_request + self.interval - now
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request = time.time()
```

作用：限制 API 请求频率。

- rpm = requests per minute，比如 100 表示每分钟最多 100 次
- interval = 60 / rpm，两次请求之间的最小间隔（秒）
- lock 保证多线程环境下不会两个线程同时通过
- wait() 计算距离上次请求是否过了足够时间，没到就 sleep

为什么需要：LLM API 通常有速率限制（429 错误），不加限流会被封。

---

## 三、LLMBots 类

### 3.1 初始化

```python
class LLMBots:
    def __init__(self, batch_size, device, api_key=None, base_url=None, model=None):
```

- batch_size — 同时跑多少局游戏（通常为 1）
- device — cpu 或 cuda
- api_key / base_url / model — LLM API 配置，支持 OpenAI 兼容接口

内部做了三件事：

1. 创建 OpenAI 客户端（兼容 DeepSeek、通义等第三方）
2. 创建限流器
3. 生成每个玩家的性格分数 theta，服从正态分布 N(mu, sigma)，值越高越倾向合作

---

### 3.2 _call_llm 方法

```python
def _call_llm(self, prompt, fallback=None):
```

单次 LLM 调用，返回 0.0 或 1.0。

流程：

1. 限流等待
2. 发送请求，system prompt 固定要求只回复 1 或 0
3. 用正则 [01] 从回复中提取第一个数字
4. 提取失败则返回 fallback（默认 0，即背叛）
5. 请求异常则指数退避重试（1s, 2s, 4s, 8s...）
6. 全部重试失败则返回 fallback

关键设计：

- temperature=0.3 低温度让输出更稳定，减少随机性
- max_tokens=5 限制回复长度，只需要一个数字
- 正则提取而非直接解析，容忍 LLM 回复"1."、"I choose 1" 等变体

---

### 3.3 _build_prompt 方法

```python
def _build_prompt(self, round_num, total_neighbors, coop_neighbors,
                  coop_rate, high_risk_edges, theta_val):
```

为单个玩家构造 prompt，返回字符串。

两种情况：

- 第 1 轮（round_num=0）：无历史，只告诉性格分数
- 第 2 轮起：告诉邻居数、合作邻居数、合作率、高风险边数、当前轮次

给 LLM 的信息模拟了人类玩家能看到的局部信息：不知道全局状态，只知道自己的邻居和上一轮结果。

---

### 3.4 decide_cooperation 方法

```python
def decide_cooperation(self, round_num, adj_matrix, prev_decisions,
                       current_capital, edge_games, delta=10.0):
```

核心方法，替代 SimulatedBots 的数学公式，用 LLM 决定每个玩家合作还是背叛。

步骤：

1. 计算每个玩家的网络统计量（邻居数、合作邻居数、合作率、高风险边数）
2. 遍历所有玩家，调用 _build_prompt 构造 prompt 列表
3. 用 ThreadPoolExecutor 并发调用所有 LLM 请求
4. 把结果列表 reshape 成 (B, N) 的 tensor 返回

参数说明：

- round_num — 当前轮次
- adj_matrix — 邻接矩阵 (B, N, N)
- prev_decisions — 上一轮所有玩家的决策 (B, N)
- current_capital — 当前资金 (B, N)
- edge_games — 每条边的风险等级 (B, N, N)，0=低风险，1=高风险
- delta — 模仿动态的温度参数（LLM 不用，但接口要兼容）

并发数由 LLMConfig.MAX_WORKERS 控制，batch_size=1, N=50 时就是 50 个并发请求。

---

### 3.5 decide_acceptance 方法

```python
def decide_acceptance(self, recommendations, prev_decisions):
    return torch.ones_like(recommendations, dtype=torch.float)
```

V2 模式下 Agent 的建议是强制采纳的（forced compliance），所以直接返回全 1。

V1 模式下这个方法需要 LLM 决定是否接受建议，但当前项目用的是 V2（MODE=1），所以空实现即可。

---

## 四、调用链路

single_game.py 中的调用顺序：

1. env = PublicGoodsGame(batch_size=1, device=device)
2. env.bots = LLMBots(...) — 替换默认的 SimulatedBots
3. env.reset() — 内部调用 bots.decide_cooperation(round=0, ...)
4. env.step(logits) — 内部调用 bots.decide_cooperation(round=N, ...)
5. 每一步 LLM 都会为 50 个玩家各发一次 API 请求

---

## 五、配置项说明（LLMConfig）

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| API_KEY | 空 | 通过 --api_key 命令行参数传入 |
| BASE_URL | None | None 用官方 OpenAI，填值可接 DeepSeek 等 |
| MODEL | mimo-v2-flash | 模型名称 |
| MAX_WORKERS | 4 | 并发线程数 |
| TEMPERATURE | 0.3 | 越低越确定，越高越随机 |
| FALLBACK | 0 | API 全部失败时默认背叛 |
| MAX_RETRIES | 5 | 最大重试次数 |
| RETRY_DELAY | 1.0 | 首次重试等待秒数，之后指数翻倍 |
| RPM | 100 | 每分钟最大请求数 |
