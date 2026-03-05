class GameConfig:
    N_PLAYERS = 16
    EPISODE_LENGTH = 15
    ERDOS_RENYI_P = 0.3
    INITIAL_CAPITAL = 1.0
    BENEFIT_B = 0.1
    COST_C = 0.05
    PENALTY_WEIGHT_P = 1.0 

class BotConfig:
    # 模拟人类参数

    
    MU_THETA = -0.304
    SIGMA_THETA = 2.410
    
    # --- 标准化统计量 ---
    MEAN_NEIGHBORS = 4.55
    STD_NEIGHBORS = 1.78
    MEAN_COOP_NEIGHBORS = 2.50
    STD_COOP_NEIGHBORS = 1.67
    MEAN_FRAC_COOP = 0.55
    STD_FRAC_COOP = 0.32

 
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
    EDGE_IN_DIM = 1
    GLOBAL_IN_DIM = 1
    
class TrainConfig:
    BATCH_SIZE = 32
    LR = 0.0004
    GAMMA = 0.99
    ENTROPY_COEF = 0.004
    VALUE_LOSS_COEF = 0.5
    MAX_EPISODES = 400000 
    LOG_INTERVAL = 10
    DEVICE = "cuda"