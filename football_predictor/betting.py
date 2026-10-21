"""Predicts a list of fixtures: probabilities, the pick, and any value bet."""

from dataclasses import dataclass, field

import numpy as np

from . import fotmob
from .config import (
    EUROPEAN_COMPETITIONS,
    KELLY_FRACTION,
    MAX_ODDS,
    MAX_STAKE,
    MIN_MATCHES,
    MIN_PROBABILITY,
)
from .markets import Bet, MatchProbabilities, candidate_bets
from .picks import blend, devig, pick, tier, tuned_settings


@dataclass
class ValueBet:
    bet: Bet
    ev: float          # blended expected value per unit staked
    stake: float       # fraction of bankroll


@dataclass
class MatchPrediction:
    fixture: fotmob.Fixture
    probs: MatchProbabilities | None = None   # the model's scoreline distribution
    p_model: np.ndarray | None = None          # model 1X2
    p_market: np.ndarray | None = None         # bookmaker 1X2, margin removed
    p_shown: np.ndarray | None = None          # what we publish: blend of the two
    p_over25: float = 0.0
    p_btts: float = 0.0
    bookmaker: str | None = None
    odds_1x2: tuple | None = None
    odds_btts: tuple | None = None             # (yes, no)
    pick_key: str = ""
    pick_label: str = ""
    pick_prob: float = 0.0
    pick_odds: float | None = None
    tier: str = ""
    value_bet: ValueBet | None = None
    data_ok: bool = True
    note: str = ""


def kelly_stake(odds, ev):
    """Fractional Kelly stake as a share of bankroll (pushes ignored)."""
    b = odds - 1
    if b <= 0 or ev <= 0:
        return 0.0
    return min(KELLY_FRACTION * ev / b, MAX_STAKE)


def best_value_bet(bets, model_weight, min_edge):
    """The single best bet that passes the value rules, if any."""
    best = None
    for bet in bets:
        if bet.odds > MAX_ODDS or bet.model_win_prob < MIN_PROBABILITY:
            continue
        ev = bet.blended_ev(model_weight)
        if ev >= min_edge and (best is None or ev > best.ev):
            # Stake on the blended view, not the (overconfident) model alone.
            best = ValueBet(bet, ev, kelly_stake(bet.odds, ev))
    return best


PICK_ODDS = {"home": ("1x2", 0), "away": ("1x2", 2), "btts_yes": ("btts", 0), "btts_no": ("btts", 1)}


def predict_fixtures(model, fixtures, with_odds=True):
    settings = tuned_settings()
    predictions = []
    for fx in fixtures:
        pred = MatchPrediction(fx)
        predictions.append(pred)
        missing = [n for i, n in ((fx.home_id, fx.home_name), (fx.away_id, fx.away_name))
                   if i is None or not model.knows(i)]
        if missing:
            pred.data_ok = False
            pred.note = f"No match history for {', '.join(missing)}"
            continue

        european = fx.league_id in EUROPEAN_COMPETITIONS
        probs = MatchProbabilities(
            model.score_matrix(fx.home_id, fx.away_id, european=european, league_id=fx.league_id))
        pred.probs = probs
        pred.p_model = np.array([probs.home, probs.draw, probs.away])
        pred.p_over25, pred.p_btts = probs.over(2.5), probs.btts

        thin = [n for i, n in ((fx.home_id, fx.home_name), (fx.away_id, fx.away_name))
                if model.matches_last_year.get(i, 0) < MIN_MATCHES]
        if thin:
            pred.data_ok = False
            pred.note = f"Little recent data for {', '.join(thin)}"

        bets = []
        if with_odds and fx.match_id is not None and not fx.started:
            markets, pred.bookmaker = fotmob.match_odds(fx.match_id)
            ftr = markets.get("Full Time Result")
            if ftr and len(ftr) == 3:
                pred.odds_1x2 = tuple(o for _, _, o in ftr)
                pred.p_market = devig(pred.odds_1x2)
            btts = {name.lower(): o for name, _, o in markets.get("Both Teams to Score", [])}
            if {"yes", "no"} <= btts.keys():
                pred.odds_btts = (btts["yes"], btts["no"])
                pred.p_btts = float(blend(pred.p_btts, devig(pred.odds_btts)[0], settings["blend_weight"]))
            bets = candidate_bets(probs, markets, fx.home_name, fx.away_name)

        pred.p_shown = blend(pred.p_model, pred.p_market, settings["blend_weight"])
        pred.pick_key, pred.pick_label, pred.pick_prob = pick(
            pred.p_shown, pred.p_over25, pred.p_btts, fx.home_name, fx.away_name)
        pred.tier = tier(pred.pick_prob)
        market, k = PICK_ODDS.get(pred.pick_key, (None, None))
        if market == "1x2" and pred.odds_1x2:
            pred.pick_odds = pred.odds_1x2[k]
        elif market == "btts" and pred.odds_btts:
            pred.pick_odds = pred.odds_btts[k]

        if pred.data_ok and bets and settings["value_enabled"]:
            pred.value_bet = best_value_bet(bets, settings["value_weight"], settings["min_edge"])
    return predictions
