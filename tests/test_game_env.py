import torch
import pytest

from config import GameConfig, BotConfig
from env.game import PublicGoodsGame


def test_reset_shapes_and_adj_properties():
    torch.manual_seed(0)
    old_n = GameConfig.N_PLAYERS
    try:
        GameConfig.N_PLAYERS = 4
        bs = 2
        device = 'cpu'
        game = PublicGoodsGame(bs, device)
        capital, prev_decisions, adj = game.reset()

        assert capital.shape == (bs, GameConfig.N_PLAYERS)
        assert prev_decisions.shape == (bs, GameConfig.N_PLAYERS)
        assert adj.shape == (bs, GameConfig.N_PLAYERS, GameConfig.N_PLAYERS)

        # adjacency symmetric and zero diagonal
        for b in range(bs):
            a = adj[b]
            assert torch.allclose(a, a.t())
            assert torch.all(a.diag() == 0)
    finally:
        GameConfig.N_PLAYERS = old_n


def test__apply_payoffs_expected():
    torch.manual_seed(0)
    old_n = GameConfig.N_PLAYERS
    try:
        GameConfig.N_PLAYERS = 3
        bs = 1
        device = 'cpu'
        game = PublicGoodsGame(bs, device)

        # set a known adjacency and cooperation decisions
        adj = torch.tensor([[[0.0,1.0,0.0],[1.0,0.0,1.0],[0.0,1.0,0.0]]], device=device)
        game.adj = adj
        game.capital = torch.ones(bs, GameConfig.N_PLAYERS, device=device) * GameConfig.INITIAL_CAPITAL

        coop_decisions = torch.tensor([[1.0, 0.0, 1.0]], device=device)
        game._apply_payoffs(coop_decisions)

        # manual expected calculation
        degree = adj.sum(dim=2).squeeze(0)
        costs = GameConfig.COST_C * degree * coop_decisions.squeeze(0)
        coop_exp = coop_decisions.unsqueeze(1).expand(-1, GameConfig.N_PLAYERS, -1)
        benefits = (adj * coop_exp).sum(dim=2).squeeze(0) * GameConfig.BENEFIT_B
        expected = torch.ones(GameConfig.N_PLAYERS, device=device) * GameConfig.INITIAL_CAPITAL + (benefits - costs)

        assert torch.allclose(game.capital.squeeze(0), expected, atol=1e-6)
    finally:
        GameConfig.N_PLAYERS = old_n


def test_step_updates_adj_and_returns_reward():
    torch.manual_seed(0)
    old_n = GameConfig.N_PLAYERS
    old_accept = BotConfig.ACCEPT_PROBS
    try:
        # use small graph for speed and set acceptance probs to deterministic 1.0
        GameConfig.N_PLAYERS = 4
        BotConfig.ACCEPT_PROBS = {(-1,0):1.0, (-1,1):1.0, (1,0):1.0, (1,1):1.0}

        bs = 2
        device = 'cpu'
        game = PublicGoodsGame(bs, device)
        capital, prev_decisions, adj_before = game.reset()

        # build logits that strongly favor 'change' (last dim)
        n = GameConfig.N_PLAYERS
        logits = torch.zeros(bs, n, n, 2, device=device)
        logits[..., 1] = 10.0

        state, reward, dist, actions_change = game.step(logits)

        capital_after, prev_decisions_after, adj_after = state

        assert reward.shape == (bs,)
        assert actions_change.shape == (bs, n, n)
        # because we forced accept probs to 1, adjacency should have changed on selected pairs
        # at least one change should be observed in upper triangle
        upper_before = torch.triu(adj_before, 1)
        upper_after = torch.triu(adj_after, 1)
        assert not torch.allclose(upper_before, upper_after)
        # current_round increments
        assert game.current_round >= 1
    finally:
        GameConfig.N_PLAYERS = old_n
        BotConfig.ACCEPT_PROBS = old_accept
