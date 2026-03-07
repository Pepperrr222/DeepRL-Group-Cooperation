import ast
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

graph_df = pd.read_csv("data/validation_graph_struct_data.csv")
coop_df = pd.read_csv("data/validation_cooperation_data.csv")

# 你可以改这里
group_id = 48
round_id = 10

sub_graph = graph_df[(graph_df["Group"] == group_id) & (graph_df["Round"] == round_id)]
sub_coop = coop_df[(coop_df["Group"] == group_id) & (coop_df["Round"] == round_id)]

G = nx.Graph()
players = sorted(sub_coop["Player"].unique().tolist())
G.add_nodes_from(players)

for _, row in sub_graph.iterrows():
    edge = row["Edge"]

    if isinstance(edge, str):
        try:
            i, j = ast.literal_eval(edge)
        except Exception:
            parts = edge.replace("(", "").replace(")", "").split(",")
            i, j = int(parts[0]), int(parts[1])
    else:
        continue

    # 用 Prev_Status=1 作为当前有边的近似重建
    if int(row["Prev_Status"]) == 1:
        G.add_edge(i, j)

# 节点颜色：上一轮/本轮合作状态
action_map = dict(zip(sub_coop["Player"], sub_coop["Cooperation"]))
capital_map = dict(zip(sub_coop["Player"], sub_coop["Capital"]))

node_colors = ["tab:blue" if action_map.get(n, 0) == 1 else "tab:red" for n in G.nodes()]
node_sizes = [300 + 300 * float(capital_map.get(n, 1.0)) for n in G.nodes()]

pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(8, 8))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=node_colors,
    node_size=node_sizes,
    font_size=9
)

plt.title(f"Group {group_id}, Round {round_id}")
plt.savefig("figs/network_snapshot_group48_round10.png", dpi=300)
plt.show()