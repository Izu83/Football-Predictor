"""Predicts a list of fixtures and picks bets."""

from dataclasses import dataclass, field

from . import fotmob
from .config import (
    EUROPEAN_COMPETITIONS,
    KELLY_FRACTION,
    MAX_ODDS,
    MAX_STAKE,
    MIN_EDGE,
    MIN_MATCHES,
    MIN_PROBABILITY,
    MODEL_WEIGHT,
)
from .markets import Bet, MatchProbabilities, candidate_bets


@dataclass
class ValueBet:
    bet: Bet
    ev: float          # blended expected value per unit staked
    stake: float       # fraction of bankroll


@dataclass
class MatchPrediction:
    fixture: fotmob.Fixture
    probs: MatchProbabilities | None = None
    bookmaker: str | None = None
    odds_1x2: tuple | None = None
    value_bets: list = field(default_factory=list)
    lean: str = ""
    note: str = ""


def kelly_stake(odds, ev):
    """Fractional Kelly stake as a share of bankroll (pushes ignored)."""
    b = odds - 1
    if b <= 0 or ev <= 0:
        return 0.0
    return min(KELLY_FRACTION * ev / b, MAX_STAKE)


def select_value_bets(bets, model_weight=MODEL_WEIGHT, min_edge=MIN_EDGE):
    picks = []
    for bet in bets:
        if bet.odds > MAX_ODDS or bet.model_win_prob < MIN_PROBABILITY:
            continue
        ev = bet.blended_ev(model_weight)
        if ev < min_edge:
            continue
        # Stake on the blended view, not the (overconfident) model alone.
        picks.append(ValueBet(bet, ev, kelly_stake(bet.odds, ev)))
    picks.sort(key=lambda v: v.ev, reverse=True)
    # One bet per market per match; correlated bets on the same match add risk, not value.
    seen, unique = set(), []
    for v in picks:
        if v.bet.market not in seen:
            seen.add(v.bet.market)
            unique.append(v)
    return unique[:2]


def lean(probs, home, away):
    """The most likely of a few simple outcomes, for matches with no value bet."""
    options = [
        (probs.home, f"{home} to win"),
        (probs.away, f"{away} to win"),
        (probs.home + probs.draw, f"{home} or draw (1X)"),
        (probs.away + probs.draw, f"{away} or draw (X2)"),
        (probs.over(2.5), "Over 2.5 goals"),
        (1 - probs.over(2.5), "Under 2.5 goals"),
        (probs.btts, "Both teams to score"),
        (1 - probs.btts, "Not both teams to score"),
    ]
    # Prefer outcomes that are likely but not near-certain (those pay almost nothing).
    p, label = max((o for o in options if o[0] < 0.85), key=lambda o: o[0])
    return f"{label} ({p:.0%}, fair odds {1 / p:.2f})"


def predict_fixtures(model, fixtures, with_odds=True):
    predictions = []
    for fx in fixtures:
        pred = MatchPrediction(fx)
        predictions.append(pred)
        missing = [n for i, n in ((fx.home_id, fx.home_name), (fx.away_id, fx.away_name))
                   if i is None or not model.knows(i)]
        if missing:
            pred.note = f"not enough data for {', '.join(missing)}"
            continue
        european = fx.league_id in EUROPEAN_COMPETITIONS
        pred.probs = MatchProbabilities(model.score_matrix(fx.home_id, fx.away_id, european=european))
        pred.lean = lean(pred.probs, fx.home_name, fx.away_name)
        thin = [n for i, n in ((fx.home_id, fx.home_name), (fx.away_id, fx.away_name))
                if model.matches_last_year.get(i, 0) < MIN_MATCHES]
        if thin:
            pred.note = f"little data for {', '.join(thin)}: no value bets"
        if not with_odds or fx.match_id is None or fx.started:
            continue
        odds_markets, pred.bookmaker = fotmob.match_odds(fx.match_id)
        ftr = odds_markets.get("Full Time Result")
        if ftr and len(ftr) == 3:
            pred.odds_1x2 = tuple(o for _, _, o in ftr)
        bets = candidate_bets(pred.probs, odds_markets, fx.home_name, fx.away_name)
        if not thin:
            pred.value_bets = select_value_bets(bets)
    return predictions
