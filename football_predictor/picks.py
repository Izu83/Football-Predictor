"""The pick shown for each match, and how confident it is.

Shared by live predictions and the backtest, so the hit rates measured in the
backtest describe exactly what the page shows.
"""

import json

import numpy as np

from .config import MIN_EDGE, MODEL_WEIGHT, REPORTS_DIR

# A pick's tier is set by its probability; the backtest measures each tier's hit rate.
TIERS = (("High", 0.72), ("Medium", 0.60), ("Low", 0.0))
MAX_PICK_PROBABILITY = 0.85  # near-certain outcomes pay almost nothing


def devig(odds):
    """Bookmaker odds -> probabilities with the margin removed proportionally."""
    inv = 1 / np.asarray(odds, dtype=float)
    return inv / inv.sum(axis=-1, keepdims=True)


def blend(model, market, weight):
    """Weighted mix of model and market probabilities (market may be None)."""
    if market is None:
        return np.asarray(model, dtype=float)
    return weight * np.asarray(model, dtype=float) + (1 - weight) * np.asarray(market, dtype=float)


def tier(prob):
    return next(name for name, low in TIERS if prob >= low)


def pick(p1x2, p_over25, p_btts, home, away):
    """Most likely simple outcome below MAX_PICK_PROBABILITY: (key, label, probability)."""
    ph, pd_, pa = p1x2
    options = [
        ("home", f"{home} to win", ph),
        ("away", f"{away} to win", pa),
        ("home_or_draw", f"{home} or draw", ph + pd_),
        ("away_or_draw", f"{away} or draw", pa + pd_),
        ("over25", "Over 2.5 goals", p_over25),
        ("under25", "Under 2.5 goals", 1 - p_over25),
        ("btts_yes", "Both teams to score", p_btts),
        ("btts_no", "Not both teams to score", 1 - p_btts),
    ]
    return max((o for o in options if o[2] < MAX_PICK_PROBABILITY), key=lambda o: o[2])


def pick_won(key, home_goals, away_goals):
    total, diff = home_goals + away_goals, home_goals - away_goals
    return {
        "home": diff > 0,
        "away": diff < 0,
        "home_or_draw": diff >= 0,
        "away_or_draw": diff <= 0,
        "over25": total > 2.5,
        "under25": total < 2.5,
        "btts_yes": home_goals > 0 and away_goals > 0,
        "btts_no": home_goals == 0 or away_goals == 0,
    }[key]


def tuned_settings():
    """Settings chosen by the last backtest on the tuning season, falling back to config.

    Value bets are switched off when even the optimistic end of their test-season
    ROI range is a loss: recommending bets that demonstrably lose is not honest.
    """
    settings = {"blend_weight": MODEL_WEIGHT, "value_weight": MODEL_WEIGHT, "min_edge": MIN_EDGE,
                "value_enabled": True}
    path = REPORTS_DIR / "backtest.json"
    if path.exists():
        bt = json.loads(path.read_text(encoding="utf-8"))
        settings.update({k: v for k, v in bt.get("chosen", {}).items() if k in settings})
        upper = (bt.get("value_bets", {}).get("roi_90") or [None, None])[1]
        settings["value_enabled"] = upper is None or upper >= 0
    return settings
