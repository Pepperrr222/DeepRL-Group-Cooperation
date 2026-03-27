为了让你的队友快速上手，你可以为他们准备一份**“项目导航指南”**。这份指南应当将复杂的论文理论（公共物品博弈、A2C、消息传递）与代码实现直接对应起来。

以下是我为你整理的文档模板，你可以直接放入项目的 `README.md` 或作为飞书/钉钉文档分享给队友。

---

# 🚀 社交规划者 (AI Social Planner) 项目指南

本项目复现了 Nature Human Behaviour 论文：*“Scaffolding cooperation in human groups with deep reinforcement learning”*。

## 1. 核心业务逻辑：我们在做什么？
我们训练了一个 **AI 社交规划者 (Agent)**，它通过观察一群人（16个玩家）的资金情况和合作倾向，每轮给出**连线（交友）或断线（拉黑）**的建议。
*   **目标**：最大化全人类的总财富（促进合作）。
*   **挑战**：玩家是自私的。如果没有 AI 干预，大家会为了个人利益选择背叛（Defect），导致群体财富缩水（公地悲剧）。

## 2. 代码地图 (Project Map)

```text
DeepRL-Group-Cooperation/
├── env/                # 【环境层】
│   ├── game.py         # 公共物品博弈逻辑，负责结算资金、处理连线建议
│   ├── bots.py         # 模拟人类决策的机器人 (基于论文拟合的参数)
│   └── llm_bots.py     # 可选：由大模型 (GPT/DeepSeek) 驱动的玩家逻辑
├── model/              # 【模型层：大脑所在地】
│   ├── graph_net.py    # GNN 底层实现 (Standard & Modified Blocks)
│   └── agent.py        # 封装 GNN，处理输入编码与策略输出
├── training/           # 【训练层】
│   ├── a2c.py          # A2C (Advantage Actor-Critic) 损失函数计算
│   └── trainer.py      # 核心训练循环：数据收集 -> 权重更新 -> 模型保存
├── planners/           # 【对照组】
│   └── baselines.py    # 包含“随机连线”、“静态网络”等用于对比的策略
├── train.py            # 启动训练主入口
├── single_game.py   # 进行一局游戏
└── ave.py      # 进行多局游戏并取平均，使用--strategy参数调整agent策略,可在config.py调整游戏局数
```

---

## 3. 图神经网络 (GNN) 深度解析
这是队友最关心的部分。我们的 Agent 是通过 **GraphNet** 思考的：

### A. 它看到了什么？ (Input)
Agent 每轮接收一个图元组 $G = (u, V, E)$：
*   **点 (V)**: 每个玩家的 `[当前资金, 上轮决策(0或1)]`。
*   **边 (E)**: 每一对玩家之间 `[当前是否有连线]`。
*   **全局 (u)**: `[当前第几轮 / 15]`。

### B. 它是怎么思考的？ (Architecture)
模型由两个 `GraphNetBlock` 组成（见 `model/graph_net.py`）：
1.  **Block 1 (感知)**: 进行第一次消息传递，让每个点知道邻居的情况，让每条边知道它连接的两个人的贫富。
2.  **Block 2 (决策)**: 
    *   **Policy Head**: 针对图中每一条边输出 `[保持, 改变]` 的概率（Logits）。
    *   **Value Head**: 针对当前局势给出一个打分（Value Estimate）。
    *   **特殊细节**: 遵循论文，Block 2 的点更新不看边信息，以实现策略与价值的解耦。

---

## 4. 强化学习流程 (A2C)
我们使用 `A2C` 算法更新 GNN 的参数：
1.  **收集阶段**：让 GNN 玩一局游戏（15轮），记录每一步的动作概率、状态预估分和实际拿到的总奖励（群体财富）。
2.  **更新阶段**：
    *   如果最终财富比预估的高（Advantage > 0），则通过梯度上升增加这组连线动作的概率。
    *   如果财富缩水，则减少对应动作概率。
    *   **掩码计算**：注意 `trainer.py` 中使用了 `triu_mask`，我们**只对上三角的有效边**计算损失，忽略对称的下三角。

---

## 5. 快速开始 (Usage)

### 1. 开始训练 (建议在服务器运行)
使用大 Batch Size (4096) 可以在 15 分钟内复现论文 $5 \times 10^7$ 步的效果：
```bash
python train.py --device cuda --batch_size 4096 --episodes 12500 --lr 0.001
```

### 2. 验证 Agent 效果
训练结束后，模型存在 `checkpoints/`。运行以下脚本看平均合作率是否达到了 70%-80%：
```bash
python evaluate_stats.py
```

### 3. 生成网络演化动画
如果想看 AI 是如何“踢走背叛者、拉拢合作者”的：
```bash
python play_visual.py
# 结果在 game_visuals/ 文件夹下
```

---


