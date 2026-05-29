# LLM 接入 Player 指南

## 现状

> [!info] 已有实现
> env/llm_bots.py 已实现 LLM 替代 Bot 玩家的决策。
> single_game.py 已有 CLI 入口。

```bash
python single_game.py --strategy graphnet --use_llm --api_key sk-xxx --model gpt-4o-mini
```

已实现 decide_cooperation（合作/背叛）和 decide_acceptance（接受/拒绝建议）。

---

## 清单

### 一、替换 Bot（玩家决策）

> [!tip] 已有基础，可直接调优

| 项目 | 文件 | 说明 |
| --- | --- | --- |
| LLM 调用封装 | env/llm_bots.py | call_llm 含限流、重试、fallback |
| API 配置 | config.py LLMConfig | key, base_url, model, temperature, RPM |
| 运行时替换 | single_game.py 第79行 | env.bots = LLMBots(...) |
| Prompt 工程 | llm_bots.py build_prompt | 核心，决定 LLM 看到什么信息、如何输出 |

V2 prompt 已包含：性格分数、邻居数、合作邻居数、合作率、高风险边数、轮次。

### 二、替换 Planner（Agent 决策）

> [!warning] 未实现

| 项目 | 说明 |
| --- | --- |
| 新建 planners/llm_planner.py | 实现 LLMPlanner，get_logits 返回 (B, N, N, 2) |
| Prompt 设计 | Planner 视角：网络拓扑、风险等级、玩家状态、收益矩阵 |
| 输出编码 | LLM 输出离散决策，编码为 logits 10/-10 或 -10/10 |
| 批量推理 | B*N*N 条边，需打包成 JSON 输出或只决策关键边 |

### 三、混合架构（可选）

- LLM 做高层策略 + 规则层执行具体边
- 全 LLM 对抗：Planner 和 Bot 各用独立 prompt

### 四、基础设施（可选）

- 每轮 prompt/response 日志，方便调试和回放
- 预估 token 成本（B*N*N*15 次调用）
- 大规模时改用 asyncio + httpx 异步化
