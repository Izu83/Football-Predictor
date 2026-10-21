"""Renders predictions for the terminal and as a static HTML page (for GitHub Pages)."""

import json
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

from .config import FEATURED_COMPETITIONS, REPORTS_DIR
from .picks import TIERS, tuned_settings

TEMPLATE_FILE = Path(__file__).with_name("page.html")


def _competition_order(pred):
    order = list(FEATURED_COMPETITIONS)
    lid = pred.fixture.league_id
    return order.index(lid) if lid in order else len(order)


def _sorted(predictions):
    return sorted(predictions, key=lambda p: (_competition_order(p), p.fixture.league_name, p.fixture.kickoff))


# ------------------------------------------------------------------ console

def print_predictions(predictions, out=print):
    if not predictions:
        out("No featured matches on this day.")
        return
    for league, preds in groupby(_sorted(predictions), key=lambda p: p.fixture.league_name):
        out(f"\n=== {league} ===")
        for p in preds:
            fx = p.fixture
            time = fx.kickoff.astimezone().strftime("%H:%M")
            out(f"\n{time}  {fx.home_name} vs {fx.away_name}" + (f"   [{fx.score}]" if fx.score else ""))
            if p.p_shown is None:
                out(f"      skipped: {p.note}")
                continue
            ph, pd_, pa = p.p_shown
            xh, xa = p.probs.expected_goals()
            i, j, _ = p.probs.most_likely_score()
            out(f"      chance  1 {ph:5.1%}   X {pd_:5.1%}   2 {pa:5.1%}   "
                f"xG {xh:.2f}-{xa:.2f}   likely score {i}-{j}   over 2.5 {p.p_over25:.0%}   BTTS {p.p_btts:.0%}")
            if p.odds_1x2:
                o = p.odds_1x2
                out(f"      {p.bookmaker or 'odds':<7} 1 {o[0]:5.2f}   X {o[1]:5.2f}   2 {o[2]:5.2f}")
            odds = f" @ {p.pick_odds:.2f}" if p.pick_odds else ""
            out(f"      pick    {p.pick_label}{odds}  ({p.pick_prob:.0%}, {p.tier} confidence)")
            if p.value_bet:
                v = p.value_bet
                out(f"      edge    {v.bet.market}: {v.bet.selection} @ {v.bet.odds:.2f}   "
                    f"expected {v.ev:+.1%}   stake {v.stake:.1%} of bankroll   (experimental)")
            if p.note:
                out(f"      note    {p.note}")


# --------------------------------------------------------------------- html

def _match_json(p):
    fx = p.fixture
    m = {
        "id": fx.match_id, "kickoff": fx.kickoff.astimezone(timezone.utc).isoformat(),
        "comp": fx.league_name, "compId": fx.league_id,
        "home": {"id": fx.home_id, "name": fx.home_name}, "away": {"id": fx.away_id, "name": fx.away_name},
        "score": fx.score or None, "started": fx.started, "finished": fx.finished,
        "dataOk": p.data_ok, "note": p.note or None,
    }
    if p.p_shown is None:
        return m
    xh, xa = p.probs.expected_goals()
    scores = sorted(((i, j, float(p.probs.m[i, j])) for i in range(6) for j in range(6)), key=lambda s: -s[2])[:5]
    m.update({
        "p": [round(float(x), 4) for x in p.p_shown],
        "pModel": [round(float(x), 4) for x in p.p_model],
        "pMarket": None if p.p_market is None else [round(float(x), 4) for x in p.p_market],
        "xg": [round(xh, 2), round(xa, 2)],
        "scores": [[i, j, round(s, 4)] for i, j, s in scores],
        "over25": round(p.p_over25, 4), "btts": round(p.p_btts, 4),
        "book": p.bookmaker, "odds": list(p.odds_1x2) if p.odds_1x2 else None,
        "pick": {"label": p.pick_label, "prob": round(p.pick_prob, 4), "tier": p.tier, "odds": p.pick_odds},
    })
    if p.value_bet:
        v = p.value_bet
        m["value"] = {"market": v.bet.market, "selection": v.bet.selection, "odds": v.bet.odds,
                      "ev": round(v.ev, 4), "stake": round(v.stake, 4)}
    return m


def _load_backtest():
    path = REPORTS_DIR / "backtest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def render_html(days, track_record=None, generated_at=None):
    """days: list of (date, predictions). Returns a complete standalone HTML page."""
    generated_at = generated_at or datetime.now(timezone.utc)
    payload = {
        "generated": generated_at.isoformat(),
        "settings": tuned_settings(),
        "tiers": [{"name": n, "min": lo} for n, lo in TIERS],
        "days": [{"date": d.isoformat(), "matches": [_match_json(p) for p in _sorted(preds)]} for d, preds in days],
        "backtest": _load_backtest(),
        "live": track_record,
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE_FILE.read_text(encoding="utf-8").replace("/*__DATA__*/null", data)
