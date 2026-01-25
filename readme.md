 1. 模型训练 (Training)
这是项目的起点，用于训练强化学习 Agent。
train.py
负责初始化环境、加载 A2C 算法并进行训练，模型会自动保存到 checkpoints/。
基本用法:
python train.py
常用参数:
# 使用 GPU，训练 10,000 局，指定随机种子
python train.py --device cuda --episodes 10000 --seed 42
 2. 单局模拟与可视化 (Single Game)
用于微观观察 Agent 的行为细节（建议连线、断线、玩家反应）。
game_runner.py (推荐)
功能：运行一局游戏。直接运行会打印 15 回合的详细数据表；被调用时返回完整数据（含邻接矩阵）。
支持策略: graphnet, static, random, coop_clustering, encouragement, neutral, max_connectivity
用法:

# 运行训练好的 AI
python game_runner.py --strategy graphnet

# 运行基于规则的“鼓励型”策略 (复现论文)
python game_runner.py --strategy encouragement
作为函数调用:
code
Python
from game_runner import run_simulation
data = run_simulation("graphnet")
# data['adjacency'] 包含每回合完整图结构
play_visual.py
功能：生成一局游戏的可视化图片。
输出: 在 game_visuals/ 目录下生成 round_01.png 到 round_15.png。
特点: 节点大小代表资金，颜色代表合作(蓝)/背叛(红)，布局固定方便观察连线变化。
用法:
code
Bash
python play_visual.py
play_demo.py
功能：在控制台打印极详细的文字日志。
特点: 双栏显示所有 16 名玩家的资金和决策，显示 Agent 的每一条建议内容及 Bot 的接受情况。
用法:
code
Bash
python play_demo.py
 3. 批量统计与基准测试 (Benchmark)
用于宏观评估策略的有效性，消除随机波动。
benchmark.py (核心评估工具)
功能：利用 GPU 并行计算，运行大规模（如 10 万次）游戏，计算精确的统计期望。
特点: 内存优化（使用累加器），速度极快。直接运行打印表格，调用时返回数据数组。
用法:
code
Bash
# 运行 100,000 次 GraphNet 策略
python benchmark.py --strategy graphnet --total 100000

# 运行 100,000 次 静态网络 (对照组)
python benchmark.py --strategy static --total 100000
作为函数调用:
code
Python
from benchmark import run_simulation
# 获取 15 回合的平均合作率曲线和资金曲线
coop_curve, cap_curve = run_simulation("graphnet", total_games=50000)
evaluate_stats.py
功能：快速并行跑 100 局游戏，输出每回合的均值和标准差。
特点: 适合训练完后快速检查模型是否收敛，或者是否有崩盘迹象。
用法:
code
Bash
python evaluate_stats.py