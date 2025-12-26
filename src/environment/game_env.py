import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class NetworkGameEnv:
    def __init__(self, num_players=16, cost=0.05, benefit=0.1):
        self.num_players = num_players
        self.cost = cost      # c = 0.05
        self.benefit = benefit # b = 0.1
        
        # 初始化一个随机图 (Erdos-Renyi)，概率为0.3
        self.graph = nx.erdos_renyi_graph(n=self.num_players, p=0.3)
        
        # 获取邻接矩阵 (Adjacency Matrix)
        self.adj_matrix = nx.to_numpy_array(self.graph)

    def update_graph(self, new_adj_matrix):
        """
        用于后续步骤：Social Planner 建议修改连边后，更新图结构
        """
        self.adj_matrix = new_adj_matrix
        self.graph = nx.from_numpy_array(new_adj_matrix)

    def calculate_payoffs(self, actions):
        """
        计算所有玩家在本回合的收益
        actions: 一个长度为 num_players 的数组，0 表示背叛，1 表示合作
        """
        actions = np.array(actions)
        
        # 计算每个玩家的度 (Degree)
        degrees = np.sum(self.adj_matrix, axis=1)
        
        # 1. 计算付出的成本 (只有合作者付出成本)
        cost_loss = -self.cost * actions * degrees
        
        # 2. 计算获得的收益 (来自邻居的贡献)
        neighbor_cooperation_count = np.dot(self.adj_matrix, actions)
        benefit_gain = self.benefit * neighbor_cooperation_count
        
        # 总收益
        payoffs = cost_loss + benefit_gain
        return payoffs

    def visualize(self, actions=None):
        """简单的可视化工具"""
        plt.figure(figsize=(6, 6))
        color_map = []
        if actions is not None:
            # 合作者蓝色，背叛者红色
            for action in actions:
                if action == 1:
                    color_map.append('skyblue') # Cooperate
                else:
                    color_map.append('lightcoral') # Defect
        else:
            color_map = 'gray'
            
        nx.draw(self.graph, node_color=color_map, with_labels=True, 
                node_size=500, font_weight='bold')
        plt.show()

# --- 测试代码 ---
if __name__ == "__main__":
    env = NetworkGameEnv(num_players=16)
    random_actions = np.random.randint(0, 2, size=16)
    print(f"玩家动作: \n{random_actions}")
    current_payoffs = env.calculate_payoffs(random_actions)
    print(f"\n玩家收益: \n{np.round(current_payoffs, 2)}")
