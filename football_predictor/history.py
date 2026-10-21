"""Live track record: every published prediction is saved before kickoff and scored afterwards.

This is the honest test of the model: predictions that were public before the
match, judged on real results. The file is committed with the site.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import fotmob
from .config import ROOT
from .markets import settle
from .picks import TIERS, pick_won

HISTORY_FILE = ROOT / "history" / "predictions.csv"
COLUMNS = [
    "match_id", "kickoff", "competition", "home", "away", "home_id", "away_id",
    "p_home", "p_draw", "p_away", "m_home", "m_draw", "m_away",
    "pick_key", "pick_label", "pick_prob", "tier", "pick_odds",
    "value_market", "value_selection", "value_kind", "value_side", "value_line", "value_odds", "value_ev",
    "home_goals", "away_goals", "status", "pick_won", "value_profit",
]


def load():
    if HISTORY_FILE.exists():
        df = pd.read_csv(HISTORY_FILE, dtype={"pick_won": object, "status": object, "value_kind": object,
                                              "value_side": object, "value_market": object, "value_selection": object})
        df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
        return df
    return pd.DataFrame(columns=COLUMNS)


def save(df):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=COLUMNS).sort_values("kickoff").to_csv(HISTORY_FILE, index=False)


def record(predictions):
    """Store the latest pre-kickoff prediction of every match (later runs overwrite earlier ones)."""
    df = load()
    rows = []
    for p in predictions:
        fx = p.fixture
        if p.p_shown is None or fx.match_id is None or fx.started:
            continue
        v = p.value_bet
        rows.append({
            "match_id": fx.match_id, "kickoff": pd.Timestamp(fx.kickoff), "competition": fx.league_name,
            "home": fx.home_name, "away": fx.away_name, "home_id": fx.home_id, "away_id": fx.away_id,
            "p_home": p.p_shown[0], "p_draw": p.p_shown[1], "p_away": p.p_shown[2],
            "m_home": p.p_market[0] if p.p_market is not None else np.nan,
            "m_draw": p.p_market[1] if p.p_market is not None else np.nan,
            "m_away": p.p_market[2] if p.p_market is not None else np.nan,
            "pick_key": p.pick_key, "pick_label": p.pick_label, "pick_prob": p.pick_prob, "tier": p.tier,
            "pick_odds": p.pick_odds,
            "value_market": v.bet.market if v else None, "value_selection": v.bet.selection if v else None,
            "value_kind": v.bet.kind if v else None, "value_side": v.bet.side if v else None,
            "value_line": v.bet.line if v else np.nan, "value_odds": v.bet.odds if v else np.nan,
            "value_ev": v.ev if v else np.nan,
            "status": "pending",
        })
    if not rows:
        return df
    new = pd.DataFrame(rows)
    pending_ids = set(df.loc[df["status"] == "pending", "match_id"])
    keep = df[~(df["match_id"].isin(new["match_id"]) & df["match_id"].isin(pending_ids))]
    new = new[~new["match_id"].isin(keep["match_id"])].reindex(columns=COLUMNS)
    df = pd.concat([keep.reindex(columns=COLUMNS), new], ignore_index=True) if len(keep) else new
    save(df)
    return df


def settle_finished(log=print):
    """Look up results for pending predictions whose match should be over."""
    df = load()
    if df.empty:
        return df
    due = df[(df["status"] == "pending") & (df["kickoff"] < datetime.now(timezone.utc) - timedelta(hours=2.5))]
    for i, row in due.iterrows():
        try:
            finished, cancelled, hg, ag = fotmob.match_result(int(row.match_id))
        except Exception as exc:
            log(f"  could not fetch result for {row.home} vs {row.away}: {exc}")
            continue
        if cancelled:
            df.loc[i, "status"] = "void"
        elif finished:
            df.loc[i, ["home_goals", "away_goals", "status"]] = [hg, ag, "settled"]
            df.loc[i, "pick_won"] = bool(pick_won(row.pick_key, hg, ag))
            if isinstance(row.value_kind, str) and row.value_kind:
                df.loc[i, "value_profit"] = settle(row.value_kind, row.value_side, float(row.value_line or 0),
                                                   float(row.value_odds), hg, ag)
    save(df)
    return df


def summary(df=None, recent=40):
    """Numbers for the page's track-record section."""
    df = load() if df is None else df
    s = df[df["status"] == "settled"].copy()
    out = {"settled": len(s), "pending": int((df["status"] == "pending").sum()), "since": None}
    if s.empty:
        return out
    out["since"] = s["kickoff"].min().strftime("%Y-%m-%d")
    s["pick_won"] = s["pick_won"].astype(str).str.lower().eq("true")
    out["pick_hit_rate"] = float(s["pick_won"].mean())
    out["pick_expected"] = float(s["pick_prob"].mean())
    out["tiers"] = []
    for name, _ in TIERS:
        g = s[s["tier"] == name]
        if len(g):
            out["tiers"].append({"tier": name, "picks": len(g), "won": float(g["pick_won"].mean()),
                                 "stated": float(g["pick_prob"].mean())})

    y = np.select([s.home_goals > s.away_goals, s.home_goals == s.away_goals], [0, 1], 2)
    shown = s[["p_home", "p_draw", "p_away"]].to_numpy(float)
    out["log_loss"] = float(-np.mean(np.log(np.clip(shown[np.arange(len(y)), y], 1e-12, None))))
    market = s[["m_home", "m_draw", "m_away"]].to_numpy(float)
    ok = ~np.isnan(market).any(axis=1)
    if ok.any():
        out["market_matches"] = int(ok.sum())
        out["log_loss_with_odds"] = float(-np.mean(np.log(shown[ok][np.arange(ok.sum()), y[ok]])))
        out["market_log_loss"] = float(-np.mean(np.log(market[ok][np.arange(ok.sum()), y[ok]])))

    v = s.dropna(subset=["value_profit"])
    if len(v):
        out["value_bets"] = len(v)
        out["value_profit"] = float(v["value_profit"].sum())
        out["value_roi"] = float(v["value_profit"].mean())

    out["recent"] = [
        {"kickoff": r.kickoff.strftime("%Y-%m-%d"), "competition": r.competition, "home": r.home, "away": r.away,
         "home_id": int(r.home_id), "away_id": int(r.away_id),
         "score": f"{int(r.home_goals)}-{int(r.away_goals)}", "pick": r.pick_label, "tier": r.tier,
         "prob": float(r.pick_prob), "won": bool(r.pick_won),
         "value": (f"{r.value_selection} @ {r.value_odds:.2f}" if isinstance(r.value_selection, str) else None),
         "value_profit": None if pd.isna(r.value_profit) else float(r.value_profit)}
        for r in s.sort_values("kickoff", ascending=False).head(recent).itertuples()
    ]
    return out
