# config.py
MODE = 1

class GameConfig_v1:
    N_PLAYERS = 16
    EPISODE_LENGTH = 15
    ERDOS_RENYI_P = 0.3
    INITIAL_CAPITAL = 1.0
    BENEFIT_B = 0.1
    COST_C = 0.05
    PENALTY_WEIGHT_P = 1.0 

class GameConfig_v2:
    # ==========================================
    # 1. 网络规模和连线概率
    # ==========================================
    N_PLAYERS = 50  # <--- 修改为 20人规模

    # 目标平均度为 4
    TARGET_AVG_DEGREE = 4.0
    
    # 虽然我们将使用 Random Regular Graph，但保留此概率用于计算 Bot 性格标准差
    # 避免方差为 0 导致归一化时除零报错 (Division by Zero)
    ERDOS_RENYI_P = TARGET_AVG_DEGREE / (N_PLAYERS - 1)
    
    EPISODE_LENGTH = 15
    INITIAL_CAPITAL = 1.0
    BENEFIT_B = 0.1
    COST_C = 0.05

    # 收益矩阵参数
    C_HIGH = 0.3
    C_LOW = 0.1
    B_HIGH = 0.8
    B_LOW = 0.2

    # 惩罚权重 (复原为原论文的设定)
    PENALTY_WEIGHT_P = 1.0 
    
    LOW_RISK_MATRIX = [
        [0.0, B_LOW],  
        [-C_LOW, B_LOW-C_LOW]  
    ]

    HIGH_RISK_MATRIX =[
        [0.0, B_HIGH],[-C_HIGH, B_HIGH-C_HIGH]  
    ]


class BotConfig:
    # 模拟人类参数
    MU_THETA = -0.304
    SIGMA_THETA = 2.410
    DELTA = 7
    # ==========================================
    # 2. 动态计算标准化统计量
    # ==========================================
    MEAN_NEIGHBORS = GameConfig_v2.TARGET_AVG_DEGREE
    # 使用 ER 图的方差近似，防止 RRG 中度数绝对固定导致 std=0
    STD_NEIGHBORS = ((GameConfig_v2.N_PLAYERS - 1) * GameConfig_v2.ERDOS_RENYI_P * (1 - GameConfig_v2.ERDOS_RENYI_P)) ** 0.5
    
    # 假设群体平均合作率维持在 50% 左右
    MEAN_COOP_NEIGHBORS = MEAN_NEIGHBORS * 0.50
    STD_COOP_NEIGHBORS = ((GameConfig_v2.N_PLAYERS - 1) * (GameConfig_v2.ERDOS_RENYI_P * 0.5) * (1 - GameConfig_v2.ERDOS_RENYI_P * 0.5)) ** 0.5
    
    MEAN_FRAC_COOP = 0.50
    STD_FRAC_COOP = 0.30

    BETA_PRIME_0 = 1.807  
    BETA_PRIME_1 = 0.818  

    BETA_0 = -0.010  
    BETA_1 = -0.75   
    BETA_2 = 1.16    
    BETA_3 = 0.46     

    # 原论文公式：Agent 建议与玩家接受率的映射表
    ACCEPT_PROBS = {
        (-1, 0): 0.774, # 建议降级低风险(断连)，对方背叛
        (-1, 1): 0.085, # 建议降级低风险(断连)，对方合作
        (1, 0): 0.287,  # 建议升级高风险(连线)，对方背叛
        (1, 1): 0.909   # 建议升级高风险(连线)，对方合作
    }

class ModelConfig:
    HIDDEN_DIM = 128
    NODE_IN_DIM = 2
    EDGE_IN_DIM = 1 if MODE == 0 else 2
    GLOBAL_IN_DIM = 1
    
class TrainConfig:
    BATCH_SIZE = 8  # 考虑到 N=20 及后续的显存占用，设为 8 非常稳妥
    LR = 0.0004
    GAMMA = 0.99
    ENTROPY_COEF = 0.004
    VALUE_LOSS_COEF = 0.5
    MAX_EPISODES = 400000 
    LOG_INTERVAL = 100
    DEVICE = "cuda"

if MODE == 0:
    GameConfig = GameConfig_v1
elif MODE == 1:
    GameConfig = GameConfig_v2
else:
    raise ValueError(f"未知的运行模式: {MODE}")