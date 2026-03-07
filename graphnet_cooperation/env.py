from dataclasses import dataclass
import random
import numpy as np

from graphnet_cooperation.bots import HumanBotPopulation


@dataclass
class GameConfig:
    n_players: int = 16
    n_rounds: int = 15
    erdos_p: float = 0.3
    init_capital: float = 1.0
    coop_benefit: float = 0.10
    coop_cost: float = 0.05
    penalty_weight: float = 1.0


class CooperationGameEnv:
    """
    One episode = one full game.
    """

    def __init__(self, config: GameConfig, seed: int = 42):
        self.cfg = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)
        self.bots = HumanBotPopulation(config.n_players, seed=seed)
        self.all_edges = [(i, j) for i in range(config.n_players) for j in range(i + 1, config.n_players)]
        self.max_edges = len(self.all_edges)
        self.reset()

    def reset(self):
        self.round_idx = 1
        self.bots.reset()

        self.capital = np.ones(self.cfg.n_players, dtype=np.float32) * self.cfg.init_capital

        self.adj = np.zeros((self.cfg.n_players, self.cfg.n_players), dtype=np.int8)
        for i, j in self.all_edges:
            if self.rng.random() < self.cfg.erdos_p:
                self.adj[i, j] = 1
                self.adj[j, i] = 1

        self.prev_actions = np.zeros(self.cfg.n_players, dtype=np.int8)
        for i in range(self.cfg.n_players):
            xs = int(self.adj[i].sum())
            self.prev_actions[i] = self.bots.sample_cooperation(
                player_id=i,
                t=1,
                xs=xs,
                xn=0,
                xr=0.0,
                capital=float(self.capital[i]),
                coop_cost=self.cfg.coop_cost,
            )

        self._apply_payoffs(self.prev_actions)
        return self.get_state()

    def _apply_payoffs(self, actions: np.ndarray):
        payoffs = np.zeros(self.cfg.n_players, dtype=np.float32)

        for i in range(self.cfg.n_players):
            neighbors = np.where(self.adj[i] == 1)[0]
            if actions[i] == 1:
                payoffs[i] -= self.cfg.coop_cost * len(neighbors)
                for j in neighbors:
                    payoffs[j] += self.cfg.coop_benefit

        self.capital += payoffs

    def get_state(self):
        degrees = self.adj.sum(axis=1).astype(np.float32)

        node_features = np.stack(
            [
                self.capital.astype(np.float32),
                self.prev_actions.astype(np.float32),
                degrees / max(1, self.cfg.n_players - 1),
            ],
            axis=1,
        )

        global_features = np.array(
            [
                self.round_idx / self.cfg.n_rounds,
                self.capital.mean(),
            ],
            dtype=np.float32,
        )

        edge_features = []
        edge_pairs = []
        for i, j in self.all_edges:
            edge_pairs.append((i, j))
            edge_features.append([float(self.adj[i, j])])
        edge_features = np.array(edge_features, dtype=np.float32)

        return {
            "node_features": node_features,
            "global_features": global_features,
            "edge_features": edge_features,
            "edge_pairs": edge_pairs,
            "adj": self.adj.copy(),
            "prev_actions": self.prev_actions.copy(),
            "capital": self.capital.copy(),
            "round_idx": self.round_idx,
        }

    def step(self, planner_actions):
        rejected_recommendations = 0
        nonzero_recommendations = 0

        for (i, j), action_binary in planner_actions.items():
            current_edge = int(self.adj[i, j])

            if current_edge == 0:
                valence = +1 if action_binary == 1 else 0
            else:
                valence = -1 if action_binary == 1 else 0

            if valence == 0:
                continue

            nonzero_recommendations += 1

            assigned_player = i if self.rng.random() < 0.5 else j
            other_player = j if assigned_player == i else i
            other_prev_action = int(self.prev_actions[other_player])

            accepted = self.bots.sample_accept(valence, other_prev_action)

            if accepted == 1:
                if valence == +1:
                    self.adj[i, j] = 1
                    self.adj[j, i] = 1
                elif valence == -1:
                    self.adj[i, j] = 0
                    self.adj[j, i] = 0
            else:
                rejected_recommendations += 1

        next_round = self.round_idx + 1
        next_actions = np.zeros(self.cfg.n_players, dtype=np.int8)

        for i in range(self.cfg.n_players):
            neighbors = np.where(self.adj[i] == 1)[0]
            xs = len(neighbors)
            if xs > 0:
                xn = int(self.prev_actions[neighbors].sum())
                xr = float(xn / xs)
            else:
                xn = 0
                xr = 0.0

            next_actions[i] = self.bots.sample_cooperation(
                player_id=i,
                t=next_round,
                xs=xs,
                xn=xn,
                xr=xr,
                capital=float(self.capital[i]),
                coop_cost=self.cfg.coop_cost,
            )

        self.prev_actions = next_actions
        self._apply_payoffs(self.prev_actions)
        self.round_idx = next_round

        penalty_rate = rejected_recommendations / self.max_edges

        cooperators = np.where(self.prev_actions == 1)[0]
        defectors = np.where(self.prev_actions == 0)[0]

        if len(cooperators) > 0:
            coop_degree = float(self.adj[cooperators].sum(axis=1).mean())
        else:
            coop_degree = 0.0

        if len(defectors) > 0:
            defect_degree = float(self.adj[defectors].sum(axis=1).mean())
        else:
            defect_degree = 0.0

        cooperator_degree_advantage = coop_degree - defect_degree

        dd_edges = 0
        for i in defectors:
            for j in defectors:
                if i < j and self.adj[i, j] == 1:
                    dd_edges += 1

        reward = float(
            self.capital.mean()
            + 0.10 * cooperator_degree_advantage
            - 0.20 * dd_edges
            - self.cfg.penalty_weight * penalty_rate
        )

        done = self.round_idx >= self.cfg.n_rounds

        info = {
            "mean_cooperation": float(self.prev_actions.mean()),
            "mean_capital": float(self.capital.mean()),
            "penalty_rate": float(penalty_rate),
            "nonzero_recommendations": int(nonzero_recommendations),
            "rejected_recommendations": int(rejected_recommendations),
            "cooperator_degree_advantage": float(cooperator_degree_advantage),
            "dd_edges": int(dd_edges),
        }

        return self.get_state(), reward, done, info