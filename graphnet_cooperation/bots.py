import math
import numpy as np


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class HumanBotPopulation:
    """
    Simulated human bots following the paper's P_cooperate and P_accept structure.
    """

    def __init__(self, n_players: int, seed: int = 42):
        self.n_players = n_players
        self.rng = np.random.default_rng(seed)

        # Supplementary Table 3
        self.mu_theta = -0.304
        self.sigma_theta = 2.410

        # cooperation model parameters
        self.beta0 = 1.807
        self.beta1 = 0.818
        self.beta2 = 0.370
        self.beta3 = 1.521

        # first-round cooperation parameters
        self.beta0_init = -0.010
        self.beta1_init = -0.193

        # acceptance probabilities
        self.phi0 = 0.774  # delete link to defector
        self.phi1 = 0.085  # delete link to cooperator
        self.phi2 = 0.287  # add link to defector
        self.phi3 = 0.909  # add link to cooperator

        self.theta = self.rng.normal(self.mu_theta, self.sigma_theta, size=n_players)

    def reset(self):
        self.theta = self.rng.normal(self.mu_theta, self.sigma_theta, size=self.n_players)

    def first_round_prob(self, player_id: int) -> float:
        z = self.beta0_init + self.beta1_init * self.theta[player_id]
        return float(np.clip(sigmoid(z), 0.001, 0.999))

    def later_round_prob(self, player_id: int, xs: float, xn: float, xr: float) -> float:
        z = self.beta0 + self.beta1 * xs + self.beta2 * xn + self.beta3 * xr + self.theta[player_id]
        return float(np.clip(sigmoid(z), 0.001, 0.999))

    def sample_cooperation(
        self,
        player_id: int,
        t: int,
        xs: float,
        xn: float,
        xr: float,
        capital: float,
        coop_cost: float,
    ) -> int:
        # capital constraint from the game rules:
        # can only cooperate if c * |N_i| <= d_i
        if coop_cost * xs > capital:
            return 0

        if t == 1:
            p = self.first_round_prob(player_id)
        else:
            p = self.later_round_prob(player_id, xs, xn, xr)

        return int(self.rng.random() < p)

    def accept_prob(self, valence: int, other_prev_action: int) -> float:
        """
        valence:
            -1 => delete link
            +1 => add link
        other_prev_action:
             0 => defected last round
             1 => cooperated last round
        """
        if valence == -1 and other_prev_action == 0:
            return self.phi0
        if valence == -1 and other_prev_action == 1:
            return self.phi1
        if valence == +1 and other_prev_action == 0:
            return self.phi2
        if valence == +1 and other_prev_action == 1:
            return self.phi3
        return 0.0

    def sample_accept(self, valence: int, other_prev_action: int) -> int:
        p = self.accept_prob(valence, other_prev_action)
        return int(self.rng.random() < p)