from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from graphnet_cooperation.env import CooperationGameEnv, GameConfig
from graphnet_cooperation.model import GraphNetPlanner


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GraphNetPlanner(node_dim=3, edge_dim=1, global_dim=2, hidden_dim=128).to(device)
model.load_state_dict(torch.load("results/graphnet_a2c_first_pass.pt", map_location=device))
model.eval()

env = CooperationGameEnv(GameConfig(), seed=0)
state = env.reset()

stats = {
    "CC": [],
    "CD": [],
    "DD": []
}

for _ in range(15):
    node_features = torch.tensor(state["node_features"], dtype=torch.float32, device=device)
    edge_features = torch.tensor(state["edge_features"], dtype=torch.float32, device=device)
    global_features = torch.tensor(state["global_features"], dtype=torch.float32, device=device)

    edge_pairs = state["edge_pairs"]
    n_nodes = node_features.shape[0]

    with torch.no_grad():
        logits, value = model(
            node_features=node_features,
            edge_features=edge_features,
            global_features=global_features,
            edge_pairs=edge_pairs,
            n_nodes=n_nodes
        )

    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # 第2列是上一轮行动：1=cooperate, 0=defect
    coop_status = state["node_features"][:, 1]

    for i, (u, v) in enumerate(edge_pairs):
        if coop_status[u] == 1 and coop_status[v] == 1:
            stats["CC"].append(probs[i])
        elif coop_status[u] != coop_status[v]:
            stats["CD"].append(probs[i])
        else:
            stats["DD"].append(probs[i])

    planner_actions = {pair: int(np.argmax(probs[i])) for i, pair in enumerate(edge_pairs)}

    state, _, done, _ = env.step(planner_actions)

    if done:
        break

print("Planner recommendation probabilities")
for k, v in stats.items():
    v = np.array(v)
    print(f"{k}: mean probs = {v.mean(axis=0)}")