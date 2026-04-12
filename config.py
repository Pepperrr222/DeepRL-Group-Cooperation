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
    # 1. 修改网络规模和连线概率
    # ==========================================
    N_PLAYERS = 100  # <--- 可以随时改为 50 或 100
    
    # 目标平均度为 4。根据公式 k = p * (N - 1) 倒推概率 p
    TARGET_AVG_DEGREE = 4.0
    ERDOS_RENYI_P = TARGET_AVG_DEGREE / (N_PLAYERS - 1)
    
    EPISODE_LENGTH = 15
    INITIAL_CAPITAL = 5.0
    BENEFIT_B = 0.1
    COST_C = 0.05

    # 收益矩阵参数保持你的设定
    C = 0.1
    B_HIGH = 0.8
    B_LOW = 0.2
    PENALTY_WEIGHT_P = 1.0 
    
    LOW_RISK_MATRIX = [[0.0, B_LOW],  
        [-C, B_LOW-C]  
    ]
    
    HIGH_RISK_MATRIX =[
        [0.0, B_HIGH],  
        [-C, B_HIGH-C]  
    ]


class BotConfig:
    # 模拟人类参数
    MU_THETA = -0.304
    SIGMA_THETA = 2.410
    
    # ==========================================
    # 2. 动态计算标准化统计量 (防止换N后Bot失常)
    # ==========================================
    # 邻居数服从二项分布 B(N-1, p)
    MEAN_NEIGHBORS = GameConfig_v2.TARGET_AVG_DEGREE
    STD_NEIGHBORS = ((GameConfig_v2.N_PLAYERS - 1) * GameConfig_v2.ERDOS_RENYI_P * (1 - GameConfig_v2.ERDOS_RENYI_P)) ** 0.5
    
    # 假设群体平均合作率维持在 50% 左右
    MEAN_COOP_NEIGHBORS = MEAN_NEIGHBORS * 0.50
    # 合作邻居数近似服从 B(N-1, p * 0.5)
    STD_COOP_NEIGHBORS = ((GameConfig_v2.N_PLAYERS - 1) * (GameConfig_v2.ERDOS_RENYI_P * 0.5) * (1 - GameConfig_v2.ERDOS_RENYI_P * 0.5)) ** 0.5
    
    MEAN_FRAC_COOP = 0.50
    STD_FRAC_COOP = 0.30

    BETA_PRIME_0 = 1.807  
    BETA_PRIME_1 = 0.818  

    BETA_0 = -0.010  
    BETA_1 = -0.75   
    BETA_2 = 1.16    
    BETA_3 = 0.46     

    ACCEPT_PROBS = {
        (-1, 0): 0.774, 
        (-1, 1): 0.085,
        (1, 0): 0.287,
        (1, 1): 0.909
    }

class ModelConfig:
    HIDDEN_DIM = 128
    NODE_IN_DIM = 2
    EDGE_IN_DIM = 1 if MODE == 0 else 2
    GLOBAL_IN_DIM = 1
    
class TrainConfig:

    BATCH_SIZE = 32  # 如果显存不够报错 CUDA OOM，请降到 16 或 8
    LR = 0.0004
    GAMMA = 0.99
    ENTROPY_COEF = 0.004
    VALUE_LOSS_COEF = 0.5
    MAX_EPISODES = 400000 
    LOG_INTERVAL = 1000
    DEVICE = "cuda"

if MODE == 0:
    GameConfig = GameConfig_v1
elif MODE == 1:
    GameConfig = GameConfig_v2
else:
    raise ValueError(f"未知的运行模式: {MODE}")