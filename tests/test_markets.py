import numpy as np
import pandas as pd
import pytest

from football_predictor.backtest import _settle_ah
from football_predictor.markets import MatchProbabilities, candidate_bets
from football_predictor.model import DixonColesModel, score_matrix


def certain(home, away, size=11):
    m = np.zeros((size, size))
    m[home, away] = 1.0
    return MatchProbabilities(m)


def test_score_matrix_is_a_distribution():
    m = score_matrix(1.6, 1.1, rho=-0.05)
    assert m.sum() == pytest.approx(1.0)
    assert (m >= 0).all()


def test_outcome_probabilities_sum_to_one():
    p = MatchProbabilities(score_matrix(1.4, 1.2, rho=-0.05))
    assert p.home + p.draw + p.away == pytest.approx(1.0)


@pytest.mark.parametrize("home,away,line,expected", [
    (1, 0, -0.5, 2.0),    # wins
    (1, 1, -0.5, 0.0),    # loses
    (1, 0, -1.0, 1.0),    # push: stake back
    (1, 0, -0.75, 1.5),   # half win, half push
    (1, 1, -0.25, 0.5),   # half lose, half push
    (1, 1, +0.25, 1.5),   # half win, half push
])
def test_asian_handicap_return(home, away, line, expected):
    assert certain(home, away).asian_handicap_return(line, 2.0, "home") == pytest.approx(expected)


@pytest.mark.parametrize("margin,line,expected", [
    (1, -0.5, 1.0), (0, -0.5, -1.0), (1, -1.0, 0.0), (1, -0.75, 0.5), (0, -0.25, -0.5),
])
def test_backtest_settlement_matches(margin, line, expected):
    assert _settle_ah(margin, line, 2.0) == pytest.approx(expected)


def test_fair_market_return_is_one_over_overround():
    p = MatchProbabilities(score_matrix(1.5, 1.0))
    bets = candidate_bets(p, {"Full Time Result": [("1", None, 2.0), ("x", None, 3.5), ("2", None, 4.0)]}, "H", "A")
    overround = 1 / 2.0 + 1 / 3.5 + 1 / 4.0
    for b in bets:
        assert b.market_return == pytest.approx(1 / overround)


def test_model_ranks_stronger_team_higher():
    rng = np.random.default_rng(0)
    rows = []
    for k in range(400):
        h, a = rng.choice(4, 2, replace=False)
        strength = {0: 2.2, 1: 1.5, 2: 1.0, 3: 0.6}
        rows.append({
            "match_id": k, "date": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=k // 2),
            "league_id": 47, "league": "Test", "home_id": h, "home": f"T{h}", "away_id": a, "away": f"T{a}",
            "home_goals": rng.poisson(strength[h] * 1.1 / strength[a] ** 0.5),
            "away_goals": rng.poisson(strength[a] / strength[h] ** 0.5),
            "european": False,
        })
    model = DixonColesModel(team_ridge=0.5).fit(pd.DataFrame(rows))
    ranking = list(model.ratings()["team_id"])
    assert ranking[0] == 0 and ranking[-1] == 3
