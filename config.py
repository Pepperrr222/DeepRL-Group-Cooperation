# config.py
class GameConfig:
    # 基础设置
    N_PLAYERS = 16
    EPISODE_LENGTH = 15
    ERDOS_RENYI_P = 0.3
    
    # 经济参数 (Paper Section A)
    INITIAL_CAPITAL = 1.0
    BENEFIT_B = 0.1
    COST_C = 0.05
    
    # 奖励函数参数 (Supp E3)
    # U_SP = (1/n * sum(d_i)) - P * (1/m * sum(changes))                               ***
    PENALTY_WEIGHT_P = 1.0 

class BotConfig:
    # 模拟人类参数 (Supp Table 3)
    MU_THETA = -0.304
    SIGMA_THETA = 2.410
    
    # 合作决策参数
    BETA_0 = 1.807
    BETA_1 = 0.818       # 邻居数量系数
  
    
    # 第一轮特殊参数
    BETA_PRIME_0 = -0.010
    BETA_PRIME_1 = -0.193
    BETA_2 = 0.370      
    BETA_3 = 1.521       
    
    # 建议接受概率 (Supp Table 3 & E4)
    # Key格式: (Recommendation, Partner_Action)
    # Recommendation: -1 (Delete), 1 (Add)
    # Partner_Action: 0 (Defect), 1 (Cooperate)
    ACCEPT_PROBS = {
        (-1, 0): 0.774, 
        (-1, 1): 0.085,
        (1, 0): 0.287,
        (1, 1): 0.909
    }

class ModelConfig:
    # 神经网络参数 (Supp Table 4)
    HIDDEN_DIM = 128
    NODE_IN_DIM = 2     # (Capital, Prev_Decision)
    EDGE_IN_DIM = 1     # (Exists?)
    GLOBAL_IN_DIM = 1   # (Time step / normalized)
    
class TrainConfig:
    # 训练参数 (Supp Table 5)
    BATCH_SIZE = 32
    LR = 0.0004
    GAMMA = 0.99
    ENTROPY_COEF = 0.004
    VALUE_LOSS_COEF = 0.5
    MAX_EPISODES = 5000 # 演示用，论文中是 5e7 steps
    LOG_INTERVAL = 10
    DEVICE = "cuda" # or "cpu"