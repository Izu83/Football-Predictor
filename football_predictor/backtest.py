"""Out-of-sample backtest.

Walk-forward: the model is refitted every week on matches played before that
week, then predicts the week. Every choice (model/bookmaker blend, value-bet
rules) is made on a tuning season and then measured once on the following test
season, which played no part in the choice.

Bookmaker odds for the big five leagues come from football-data.co.uk, matched
to FotMob matches by date, team names and score.
"""

import difflib
import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import REPORTS_DIR
from .data import normalize_name, played
from .markets import MatchProbabilities
from .model import DixonColesModel
from .net import get_text
from .picks import TIERS, blend, devig, pick, pick_won, tier

FOOTBALL_DATA = {"E0": 47, "SP1": 87, "I1": 55, "D1": 54, "F1": 53}
TUNE = ("2024-08-01", "2025-06-01", "2425")
TEST = ("2025-08-01", "2026-06-01", "2526")
BLEND_WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)
VALUE_GRID = [(w, e) for w in (0.25, 0.35, 0.5, 0.75, 1.0) for e in (0.03, 0.05, 0.08, 0.12)]
MIN_TUNE_BETS = 100


def walk_forward(results, start, end, step_days=7, **model_kwargs):
    """Predictions for every played match in [start, end), each made before its week began."""
    games = played(results)
    rows = []
    cut = pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end, tz="UTC")
    while cut < end:
        nxt = cut + pd.Timedelta(days=step_days)
        week = games[(games["date"] >= cut) & (games["date"] < nxt)]
        if len(week):
            model = DixonColesModel(**model_kwargs).fit(results, as_of=cut)
            for g in week.itertuples():
                if not (model.knows(g.home_id) and model.knows(g.away_id)):
                    continue
                probs = MatchProbabilities(
                    model.score_matrix(g.home_id, g.away_id, european=g.european, league_id=g.league_id))
                rows.append({
                    "match_id": g.match_id, "date": g.date, "league_id": g.league_id, "league": g.league,
                    "home": g.home, "away": g.away, "home_goals": g.home_goals, "away_goals": g.away_goals,
                    "p_home": probs.home, "p_draw": probs.draw, "p_away": probs.away,
                    "p_over25": probs.over(2.5), "p_btts": probs.btts,
                    "probs": probs,  # kept for handicap settlement
                })
        cut = nxt
    return pd.DataFrame(rows)


def outcome_index(df):
    return np.select([df.home_goals > df.away_goals, df.home_goals == df.away_goals], [0, 1], 2)


def log_loss(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))


def brier(p, y):
    return float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))


# --------------------------------------------------------------- odds data

def load_football_data(season_code):
    frames = []
    for div, league_id in FOOTBALL_DATA.items():
        text = get_text(f"https://www.football-data.co.uk/mmz4281/{season_code}/{div}.csv",
                        max_age=float("inf"), suffix=".csv")
        df = pd.read_csv(io.StringIO(text.lstrip("﻿")))
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG"])
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df["league_id"] = league_id
        # itertuples can't expose column names containing '>' or '<'.
        frames.append(df.rename(columns={"B365>2.5": "B365O25", "B365<2.5": "B365U25"}))
    return pd.concat(frames, ignore_index=True)


def _similarity(a, b):
    a, b = normalize_name(a), normalize_name(b)
    score = difflib.SequenceMatcher(None, a, b).ratio()
    if a and b and (a in b or b in a):
        score = max(score, 0.9)
    return score


def attach_odds(preds, fd):
    """Join football-data odds onto predictions by league, date (+-1 day), team names and score."""
    preds = preds.copy()
    preds["day"] = preds["date"].dt.tz_convert(None).dt.normalize()
    matched = {}
    for row in fd.itertuples():
        cands = preds[(preds.league_id == row.league_id)
                      & ((preds.day - row.date).abs() <= pd.Timedelta(days=1))
                      & (preds.home_goals == row.FTHG) & (preds.away_goals == row.FTAG)]
        best, best_score = None, 0.0
        for c in cands.itertuples():
            s = _similarity(row.HomeTeam, c.home) + _similarity(row.AwayTeam, c.away)
            if s > best_score:
                best, best_score = c.Index, s
        if best is not None and best_score >= 0.8:
            matched[best] = row
    cols = {
        "B365H": "o_home", "B365D": "o_draw", "B365A": "o_away",
        "PSCH": "c_home", "PSCD": "c_draw", "PSCA": "c_away",
        "B365O25": "o_over25", "B365U25": "o_under25",
        "AHh": "ah_line", "B365AHH": "o_ah_home", "B365AHA": "o_ah_away",
    }
    for src, dst in cols.items():
        preds[dst] = [getattr(matched[i], src, np.nan) if i in matched else np.nan for i in preds.index]
    return preds.dropna(subset=["o_home", "o_draw", "o_away"]), len(matched), len(fd)


# ------------------------------------------------------------------ betting

@dataclass
class BetResult:
    market: str
    odds: float
    ev: float
    profit: float  # per unit staked


def _settle_ah(margin, line, odds):
    if abs(line * 4) % 2 == 1:
        return 0.5 * (_settle_ah(margin, line - 0.25, odds) + _settle_ah(margin, line + 0.25, odds))
    x = margin + line
    return odds - 1 if x > 0 else 0.0 if x == 0 else -1.0


def simulate_bets(preds, model_weight, min_edge, min_prob=0.25, max_odds=5.0):
    """Flat 1-unit stakes on every bet that passes the value rules, settled on real results."""
    out = []
    for r in preds.itertuples():
        y = 0 if r.home_goals > r.away_goals else 1 if r.home_goals == r.away_goals else 2
        odds = np.array([r.o_home, r.o_draw, r.o_away])
        fair = devig(odds)
        model = np.array([r.p_home, r.p_draw, r.p_away])
        candidates = []
        for k in range(3):
            ev = model_weight * (model[k] * odds[k] - 1) + (1 - model_weight) * (fair[k] * odds[k] - 1)
            candidates.append(("1X2", odds[k], model[k], ev, odds[k] - 1 if k == y else -1.0))

        if not np.isnan(r.o_over25) and not np.isnan(r.o_under25):
            ou = np.array([r.o_over25, r.o_under25])
            fair_ou = devig(ou)
            p_ou = np.array([r.p_over25, 1 - r.p_over25])
            went_over = r.home_goals + r.away_goals > 2.5
            for k in range(2):
                ev = model_weight * (p_ou[k] * ou[k] - 1) + (1 - model_weight) * (fair_ou[k] * ou[k] - 1)
                won = went_over if k == 0 else not went_over
                candidates.append(("O/U 2.5", ou[k], p_ou[k], ev, ou[k] - 1 if won else -1.0))

        if not np.isnan(r.ah_line) and not np.isnan(r.o_ah_home):
            ah = np.array([r.o_ah_home, r.o_ah_away])
            fair_ah = devig(ah)
            diff = r.home_goals - r.away_goals
            for k, (side, line) in enumerate((("home", r.ah_line), ("away", -r.ah_line))):
                ret = r.probs.asian_handicap_return(line, ah[k], side)
                ev = model_weight * (ret - 1) + (1 - model_weight) * (fair_ah[k] * ah[k] - 1)
                p_win = r.probs.asian_handicap_win_prob(line, side)
                profit = _settle_ah(diff if side == "home" else -diff, line, ah[k])
                candidates.append(("Asian Handicap", ah[k], p_win, ev, profit))

        best_per_market = {}
        for market, o, p, ev, profit in candidates:
            if o > max_odds or p < min_prob or ev < min_edge:
                continue
            if market not in best_per_market or ev > best_per_market[market].ev:
                best_per_market[market] = BetResult(market, o, ev, profit)
        out.extend(best_per_market.values())
    return pd.DataFrame([b.__dict__ for b in out], columns=["market", "odds", "ev", "profit"])


def bootstrap_roi(profits, n=2000, seed=0):
    """90% interval for ROI: how much of the result could be luck."""
    if len(profits) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    p = np.asarray(profits)
    means = rng.choice(p, size=(n, len(p))).mean(axis=1)
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


# ----------------------------------------------------------------- analysis

def displayed(preds, weight):
    """The probabilities the page would show: blended with bet365 where odds exist."""
    p_model = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    if "o_home" not in preds:
        return p_model, preds["p_over25"].to_numpy()
    p = blend(p_model, devig(preds[["o_home", "o_draw", "o_away"]].to_numpy()), weight)
    over = preds["p_over25"].to_numpy().copy()
    ou = preds[["o_over25", "o_under25"]].to_numpy()
    ok = ~np.isnan(ou).any(axis=1)
    over[ok] = blend(over[ok], devig(ou[ok])[:, 0], weight)
    return p, over


def pick_record(preds, p1x2, p_over25):
    """Hit rate of the page's pick, by confidence tier."""
    rows = []
    for r, p, over in zip(preds.itertuples(), p1x2, p_over25):
        key, _, prob = pick(p, over, r.p_btts, r.home, r.away)
        rows.append((tier(prob), prob, pick_won(key, r.home_goals, r.away_goals)))
    df = pd.DataFrame(rows, columns=["tier", "prob", "won"])
    out = []
    for name, _ in TIERS:
        g = df[df.tier == name]
        if len(g):
            out.append({"tier": name, "picks": len(g), "stated": float(g.prob.mean()), "won": float(g.won.mean())})
    return out


def calibration(p, y, bins=10):
    """Reliability table for 1X2 probabilities: stated vs actual frequency."""
    probs = p.ravel()
    hits = (np.eye(3)[y]).ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 30:
            out.append({"stated": float(probs[m].mean()), "actual": float(hits[m].mean()), "n": int(m.sum())})
    return out


def season(results, period, model_kwargs, log):
    start, end, code = period
    log(f"  walk-forward {start} to {end}...")
    preds = walk_forward(results, start, end, **model_kwargs)
    big5, n_matched, n_total = attach_odds(preds[preds.league_id.isin(FOOTBALL_DATA.values())],
                                          load_football_data(code))
    log(f"  {len(preds):,} matches predicted, {n_matched}/{n_total} big-five matches matched to odds")
    return preds, big5


def run(results, model_kwargs=None, log=print):
    model_kwargs = model_kwargs or {}

    log("Tuning season (2024/25): choosing settings")
    tune_all, tune_big5 = season(results, TUNE, model_kwargs, log)
    y = outcome_index(tune_big5)
    p_model = tune_big5[["p_home", "p_draw", "p_away"]].to_numpy()
    p_b365 = devig(tune_big5[["o_home", "o_draw", "o_away"]].to_numpy())
    blend_losses = {float(w): log_loss(blend(p_model, p_b365, w), y) for w in BLEND_WEIGHTS}
    blend_weight = min(blend_losses, key=blend_losses.get)
    log(f"  best blend: {blend_weight:.2f} model + {1 - blend_weight:.2f} bet365 "
        f"(log loss {blend_losses[blend_weight]:.4f}; bet365 alone {blend_losses[0.0]:.4f}, "
        f"model alone {blend_losses[1.0]:.4f})")

    best = None
    for w, e in VALUE_GRID:
        bets = simulate_bets(tune_big5, w, e)
        if len(bets) >= MIN_TUNE_BETS:
            roi = bets.profit.mean()
            if best is None or roi > best[2]:
                best = (w, e, roi, len(bets))
    value_weight, min_edge, tune_roi, tune_bets = best
    log(f"  best value-bet rule: weight {value_weight}, min edge {min_edge:.0%} "
        f"-> {tune_bets} bets, ROI {tune_roi:+.1%} on the tuning season")

    log("\nTest season (2025/26): measuring, nothing chosen here")
    test_all, test_big5 = season(results, TEST, model_kwargs, log)

    y_all = outcome_index(test_all)
    p_all = test_all[["p_home", "p_draw", "p_away"]].to_numpy()
    base = np.tile(np.bincount(y_all, minlength=3) / len(y_all), (len(y_all), 1))

    y5 = outcome_index(test_big5)
    p5_model = test_big5[["p_home", "p_draw", "p_away"]].to_numpy()
    p5_b365 = devig(test_big5[["o_home", "o_draw", "o_away"]].to_numpy())
    p5_shown, over5_shown = displayed(test_big5, blend_weight)
    closing = test_big5[["c_home", "c_draw", "c_away"]].to_numpy()
    ok = ~np.isnan(closing).any(axis=1)

    accuracy = {
        "matches": len(test_all),
        "model_log_loss": log_loss(p_all, y_all),
        "base_rate_log_loss": log_loss(base, y_all),
        "model_favourite_won": float(np.mean(p_all.argmax(1) == y_all)),
        "big5_matches": len(test_big5),
        "big5_model_log_loss": log_loss(p5_model, y5),
        "big5_bet365_log_loss": log_loss(p5_b365, y5),
        "big5_shown_log_loss": log_loss(p5_shown, y5),
        "big5_pinnacle_closing_log_loss": log_loss(devig(closing[ok]), y5[ok]) if ok.any() else None,
        "big5_pinnacle_matches": int(ok.sum()),
    }
    log(f"  all competitions: model log loss {accuracy['model_log_loss']:.4f} "
        f"(always-average {accuracy['base_rate_log_loss']:.4f}), favourite won {accuracy['model_favourite_won']:.1%}")
    log(f"  big five: model {accuracy['big5_model_log_loss']:.4f}, bet365 {accuracy['big5_bet365_log_loss']:.4f}, "
        f"shown (blend) {accuracy['big5_shown_log_loss']:.4f}, Pinnacle closing {accuracy['big5_pinnacle_closing_log_loss']:.4f}")

    by_competition = []
    for (lid, name), idx in test_all.groupby(["league_id", "league"]).groups.items():
        m = test_all.index.isin(idx)
        if m.sum() >= 100:
            by_competition.append({"competition": name, "matches": int(m.sum()),
                                   "log_loss": log_loss(p_all[m], y_all[m]),
                                   "favourite_won": float(np.mean(p_all[m].argmax(1) == y_all[m]))})
    by_competition.sort(key=lambda c: -c["matches"])

    picks_with_odds = pick_record(test_big5, p5_shown, over5_shown)
    picks_model_only = pick_record(test_all, p_all, test_all["p_over25"].to_numpy())
    for label, rec in (("with odds", picks_with_odds), ("model only", picks_model_only)):
        log(f"  picks {label}: " + ", ".join(f"{t['tier']} {t['won']:.0%} of {t['picks']} (stated {t['stated']:.0%})" for t in rec))

    test_bets = simulate_bets(test_big5, value_weight, min_edge)
    lo, hi = bootstrap_roi(test_bets.profit)
    value = {
        "bets": len(test_bets),
        "won": float((test_bets.profit > 0).mean()) if len(test_bets) else None,
        "roi": float(test_bets.profit.mean()) if len(test_bets) else None,
        "roi_90": [lo, hi],
        "tune_roi": float(tune_roi),
        "tune_bets": int(tune_bets),
    }
    if len(test_bets):
        log(f"  value bets: {len(test_bets)} bets, ROI {value['roi']:+.1%} (90% range {lo:+.1%} to {hi:+.1%})")

    summary = {
        "tune_period": f"{TUNE[0]} to {TUNE[1]}",
        "test_period": f"{TEST[0]} to {TEST[1]}",
        "model": {k: v for k, v in model_kwargs.items()},
        "chosen": {"blend_weight": blend_weight, "value_weight": value_weight, "min_edge": min_edge},
        "accuracy": accuracy,
        "by_competition": by_competition,
        "picks_with_odds": picks_with_odds,
        "picks_model_only": picks_model_only,
        "calibration": calibration(p_all, y_all),
        "value_bets": value,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "backtest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"\nSaved {REPORTS_DIR / 'backtest.json'}")
    return summary


def tune_model(results, grid, log=print):
    """Pick model parameters on the tuning season only (log loss over all competitions)."""
    best = None
    for params in grid:
        preds = walk_forward(results, TUNE[0], TUNE[1], **params)
        y = outcome_index(preds)
        ll = log_loss(preds[["p_home", "p_draw", "p_away"]].to_numpy(), y)
        log(f"  {params}: log loss {ll:.4f} on {len(preds):,} matches")
        if best is None or ll < best[0]:
            best = (ll, params)
    log(f"Best: {best[1]}")
    return best[1]
