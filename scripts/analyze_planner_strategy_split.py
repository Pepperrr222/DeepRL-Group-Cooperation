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
    "CC_add": [],
    "CC_delete": [],
    "CD_add": [],
    "CD_delete": [],
    "DD_add": [],
    "DD_delete": [],
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
    coop_status = state["node_features"][:, 1]   # 上一轮是否合作
    adj = state["adj"]

    for i, (u, v) in enumerate(edge_pairs):
        has_edge = int(adj[u, v])

        if coop_status[u] == 1 and coop_status[v] == 1:
            pair_type = "CC"
        elif coop_status[u] != coop_status[v]:
            pair_type = "CD"
        else:
            pair_type = "DD"

        if has_edge == 1:
            stats[f"{pair_type}_delete"].append(probs[i, 1])  # action 1 = 删边
        else:
            stats[f"{pair_type}_add"].append(probs[i, 1])     # action 1 = 加边

    planner_actions = {pair: int(np.argmax(probs[i])) for i, pair in enumerate(edge_pairs)}
    state, _, done, _ = env.step(planner_actions)
    if done:
        break

print("Planner action-1 probability by pair type and edge status")
for k, v in stats.items():
    if len(v) == 0:
        print(f"{k}: no samples")
    else:
        print(f"{k}: mean={np.mean(v):.4f}, n={len(v)}")