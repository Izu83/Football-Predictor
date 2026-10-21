"""Walk-forward backtest: refit every week on past data only, predict the next week.

Odds and closing prices for the big five leagues come from football-data.co.uk,
matched to FotMob matches by date and team names.
"""

import difflib
import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EUROPEAN_COMPETITIONS, MIN_EDGE, MODEL_WEIGHT, REPORTS_DIR
from .data import normalize_name, played
from .markets import MatchProbabilities
from .model import DixonColesModel
from .net import get_text

FOOTBALL_DATA = {"E0": 47, "SP1": 87, "I1": 55, "D1": 54, "F1": 53}


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
                probs = MatchProbabilities(model.score_matrix(g.home_id, g.away_id, european=g.european))
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
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


# --------------------------------------------------------------- odds data

def load_football_data(season_code="2526"):
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
    """Join football-data odds onto predictions by league, date (+-1 day) and team names."""
    preds = preds.copy()
    preds["day"] = preds["date"].dt.tz_convert(None).dt.normalize()
    matched = {}
    for row in fd.itertuples():
        cands = preds[(preds.league_id == row.league_id)
                      & ((preds.day - row.date).abs() <= pd.Timedelta(days=1))]
        best, best_score = None, 0.0
        for c in cands.itertuples():
            s = _similarity(row.HomeTeam, c.home) + _similarity(row.AwayTeam, c.away)
            # Scores must agree too; that rules out almost every false match.
            if (c.home_goals, c.away_goals) != (row.FTHG, row.FTAG):
                continue
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
    return preds, len(matched), len(fd)


def devig(o):
    inv = 1 / o
    return inv / inv.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ betting

@dataclass
class BetResult:
    market: str
    odds: float
    ev: float
    profit: float  # per unit staked


def simulate_bets(preds, model_weight, min_edge, min_prob=0.25, max_odds=5.0):
    """Flat 1-unit stakes on every bet that passes the value rules, settled on real results."""
    out = []
    for r in preds.dropna(subset=["o_home"]).itertuples():
        y = 0 if r.home_goals > r.away_goals else 1 if r.home_goals == r.away_goals else 2
        odds = np.array([r.o_home, r.o_draw, r.o_away])
        fair = devig(odds[None])[0]
        model = np.array([r.p_home, r.p_draw, r.p_away])
        candidates = []
        for k in range(3):
            ev = model_weight * (model[k] * odds[k] - 1) + (1 - model_weight) * (fair[k] * odds[k] - 1)
            candidates.append(("1X2", odds[k], model[k], ev, odds[k] - 1 if k == y else -1.0))

        if not np.isnan(r.o_over25) and not np.isnan(r.o_under25):
            ou = np.array([r.o_over25, r.o_under25])
            fair_ou = devig(ou[None])[0]
            p_ou = np.array([r.p_over25, 1 - r.p_over25])
            went_over = r.home_goals + r.away_goals > 2.5
            for k in range(2):
                ev = model_weight * (p_ou[k] * ou[k] - 1) + (1 - model_weight) * (fair_ou[k] * ou[k] - 1)
                won = went_over if k == 0 else not went_over
                candidates.append(("O/U 2.5", ou[k], p_ou[k], ev, ou[k] - 1 if won else -1.0))

        if not np.isnan(r.ah_line) and not np.isnan(r.o_ah_home):
            ah = np.array([r.o_ah_home, r.o_ah_away])
            fair_ah = devig(ah[None])[0]
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
    return pd.DataFrame([b.__dict__ for b in out])


def _settle_ah(margin, line, odds):
    if abs(line * 4) % 2 == 1:
        return 0.5 * (_settle_ah(margin, line - 0.25, odds) + _settle_ah(margin, line + 0.25, odds))
    x = margin + line
    return odds - 1 if x > 0 else 0.0 if x == 0 else -1.0


# -------------------------------------------------------------------- report

def evaluate(preds, log=print):
    y = outcome_index(preds)
    p = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    base = np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1))
    log(f"  matches: {len(preds)}")
    log(f"  model     log loss {log_loss(p, y):.4f}  brier {brier(p, y):.4f}  accuracy {np.mean(p.argmax(1) == y):.1%}")
    log(f"  base rate log loss {log_loss(base, y):.4f}  brier {brier(base, y):.4f}")
    return {"matches": len(preds), "log_loss": log_loss(p, y), "base_rate_log_loss": log_loss(base, y),
            "accuracy": float(np.mean(p.argmax(1) == y))}


def market_comparison(preds, log=print):
    """Model vs bookmaker on matches with odds, and the best blend of the two."""
    df = preds.dropna(subset=["o_home", "o_draw", "o_away"])
    y = outcome_index(df)
    p_model = df[["p_home", "p_draw", "p_away"]].to_numpy()
    p_b365 = devig(df[["o_home", "o_draw", "o_away"]].to_numpy())
    log(f"  matches with odds: {len(df)}")
    log(f"  model             log loss {log_loss(p_model, y):.4f}  brier {brier(p_model, y):.4f}")
    log(f"  bet365 (no vig)   log loss {log_loss(p_b365, y):.4f}  brier {brier(p_b365, y):.4f}")
    summary = {"matches": len(df), "model_log_loss": log_loss(p_model, y), "bet365_log_loss": log_loss(p_b365, y),
               "model_accuracy": float(np.mean(p_model.argmax(1) == y)),
               "bet365_accuracy": float(np.mean(p_b365.argmax(1) == y))}
    closing = df[["c_home", "c_draw", "c_away"]].to_numpy()
    ok = ~np.isnan(closing).any(axis=1)
    if ok.any():
        log(f"  Pinnacle closing  log loss {log_loss(devig(closing[ok]), y[ok]):.4f}  ({ok.sum()} matches)")
        summary["pinnacle_closing_log_loss"] = log_loss(devig(closing[ok]), y[ok])
    best_w, best_ll = 0.0, log_loss(p_b365, y)
    for w in np.arange(0.05, 1.0001, 0.05):
        ll = log_loss(w * p_model + (1 - w) * p_b365, y)
        if ll < best_ll:
            best_w, best_ll = float(w), ll
    log(f"  best blend: {best_w:.2f} x model + {1 - best_w:.2f} x bet365 -> log loss {best_ll:.4f}")
    summary["best_model_weight"] = best_w
    return summary


def run(results, start="2025-08-01", end="2026-06-01", grid=None, log=print):
    grid = grid or [{}]
    scores = []
    for params in grid:
        preds = walk_forward(results, start, end, **params)
        log(f"\nParameters {params or 'default'}")
        scores.append((evaluate(preds, log), params, preds))
    overall, best_params, preds = min(scores, key=lambda s: s[0]["log_loss"])
    if len(grid) > 1:
        log(f"\nBest parameters: {best_params}")

    log("\nBy competition (model log loss):")
    y = outcome_index(preds)
    p = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    by_competition = []
    for (lid, name), idx in preds.groupby(["league_id", "league"]).groups.items():
        mask = preds.index.isin(idx)
        if lid in EUROPEAN_COMPETITIONS or lid in FOOTBALL_DATA.values():
            ll, acc = log_loss(p[mask], y[mask]), float(np.mean(p[mask].argmax(1) == y[mask]))
            log(f"  {name:<22} {mask.sum():>4} matches  log loss {ll:.4f}  accuracy {acc:.1%}")
            by_competition.append({"competition": name, "matches": int(mask.sum()), "log_loss": ll, "accuracy": acc})

    log("\nAgainst the bookmakers (big five leagues, 2025/26):")
    fd = load_football_data("2526")
    big5 = preds[preds.league_id.isin(FOOTBALL_DATA.values())]
    big5, n_matched, n_total = attach_odds(big5, fd)
    log(f"  matched {n_matched}/{n_total} football-data rows to FotMob matches")
    vs_market = market_comparison(big5, log)

    log("\nValue-bet simulation (bet365 pre-match odds, flat stakes):")
    simulations = []
    for w in sorted({vs_market["best_model_weight"], MODEL_WEIGHT, 0.5, 1.0}):
        for edge in sorted({0.03, MIN_EDGE, 0.10}):
            bets = simulate_bets(big5, w, edge)
            if bets.empty:
                log(f"  weight {w:.2f} min edge {edge:.0%}: no bets")
                continue
            roi = bets.profit.sum() / len(bets)
            hits = (bets.profit > 0).mean()
            by_market = ", ".join(f"{m} {g.profit.sum() / len(g):+.1%} ({len(g)})" for m, g in bets.groupby("market"))
            log(f"  weight {w:.2f} min edge {edge:.0%}: {len(bets):>4} bets, won {hits:.0%}, ROI {roi:+.1%}  [{by_market}]")
            simulations.append({"model_weight": w, "min_edge": edge, "bets": len(bets),
                                "won": float(hits), "roi": float(roi),
                                "current_settings": bool(np.isclose(w, MODEL_WEIGHT) and np.isclose(edge, MIN_EDGE))})

    summary = {
        "period": f"{start} to {end}",
        "parameters": best_params,
        "overall": overall,
        "by_competition": by_competition,
        "vs_market": vs_market,
        "value_bets": simulations,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "backtest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"\nSaved {REPORTS_DIR / 'backtest.json'}")
    return summary
