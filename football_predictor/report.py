"""Renders predictions for the terminal and as a static HTML page (for GitHub Pages)."""

import html
import json
from datetime import datetime, timezone
from itertools import groupby

from .config import FEATURED_COMPETITIONS, MIN_EDGE, MODEL_WEIGHT, REPORTS_DIR


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
            if p.probs is None:
                out(f"      skipped: {p.note}")
                continue
            pr = p.probs
            xh, xa = pr.expected_goals()
            i, j, _ = pr.most_likely_score()
            out(f"      model   1 {pr.home:5.1%}   X {pr.draw:5.1%}   2 {pr.away:5.1%}   "
                f"xG {xh:.2f}-{xa:.2f}   likely score {i}-{j}   over 2.5 {pr.over(2.5):.0%}   BTTS {pr.btts:.0%}")
            if p.odds_1x2:
                o = p.odds_1x2
                out(f"      {p.bookmaker or 'odds':<7} 1 {o[0]:5.2f}   X {o[1]:5.2f}   2 {o[2]:5.2f}   "
                    f"(fair from model: {1 / pr.home:.2f} / {1 / pr.draw:.2f} / {1 / pr.away:.2f})")
            if p.value_bets:
                for v in p.value_bets:
                    out(f"      VALUE   {v.bet.market}: {v.bet.selection} @ {v.bet.odds:.2f}   "
                        f"edge {v.ev:+.1%}   stake {v.stake:.1%} of bankroll")
            out(f"      lean    {p.lean}")
            if p.note:
                out(f"      note    {p.note}")


# --------------------------------------------------------------------- html

def _load_backtest():
    path = REPORTS_DIR / "backtest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _match_card(p):
    fx = p.fixture
    e = html.escape
    kickoff = fx.kickoff.astimezone(timezone.utc).isoformat()
    head = (f'<div class="teams"><span>{e(fx.home_name)}</span>'
            f'<span class="vs">{e(fx.score) if fx.score else "vs"}</span>'
            f'<span>{e(fx.away_name)}</span></div>')
    time = f'<time datetime="{kickoff}">{fx.kickoff.strftime("%H:%M")} UTC</time>'
    if p.probs is None:
        return f'<article class="match muted">{time}{head}<p class="note">Skipped: {e(p.note)}</p></article>'

    pr = p.probs
    xh, xa = pr.expected_goals()
    i, j, _ = pr.most_likely_score()
    bar = (f'<div class="bar" role="img" aria-label="Home {pr.home:.0%}, draw {pr.draw:.0%}, away {pr.away:.0%}">'
           f'<span class="h" style="width:{pr.home * 100:.1f}%">{pr.home:.0%}</span>'
           f'<span class="d" style="width:{pr.draw * 100:.1f}%">{pr.draw:.0%}</span>'
           f'<span class="a" style="width:{pr.away * 100:.1f}%">{pr.away:.0%}</span></div>')
    stats = (f'<dl class="stats"><div><dt>xG</dt><dd>{xh:.2f} – {xa:.2f}</dd></div>'
             f'<div><dt>Likely score</dt><dd>{i}–{j}</dd></div>'
             f'<div><dt>Over 2.5</dt><dd>{pr.over(2.5):.0%}</dd></div>'
             f'<div><dt>BTTS</dt><dd>{pr.btts:.0%}</dd></div></dl>')
    odds = ""
    if p.odds_1x2:
        o = p.odds_1x2
        odds = (f'<table class="odds"><tr><th></th><th>1</th><th>X</th><th>2</th></tr>'
                f'<tr><td>{e(p.bookmaker or "Odds")}</td><td>{o[0]:.2f}</td><td>{o[1]:.2f}</td><td>{o[2]:.2f}</td></tr>'
                f'<tr><td>Model fair</td><td>{1 / pr.home:.2f}</td><td>{1 / pr.draw:.2f}</td><td>{1 / pr.away:.2f}</td></tr></table>')
    picks = "".join(
        f'<p class="pick value"><b>Value:</b> {e(v.bet.market)} – {e(v.bet.selection)} @ {v.bet.odds:.2f} '
        f'<span class="tag">edge {v.ev:+.1%} · stake {v.stake:.1%}</span></p>'
        for v in p.value_bets
    )
    lean = f'<p class="pick"><b>Lean:</b> {e(p.lean)}</p>'
    note = f'<p class="note">{e(p.note[:1].upper() + p.note[1:])}.</p>' if p.note else ""
    return f'<article class="match">{time}{head}{bar}{stats}{odds}{picks}{lean}{note}</article>'


def _backtest_section(bt):
    if not bt:
        return ""
    vm = bt["vs_market"]
    current = next((s for s in bt["value_bets"] if s["current_settings"]), None)
    rows = "".join(
        f'<tr><td>{html.escape(c["competition"])}</td><td>{c["matches"]}</td><td>{c["accuracy"]:.0%}</td></tr>'
        for c in bt["by_competition"]
    )
    roi = (f'<p>Value bets with the current settings ({MODEL_WEIGHT:.0%} model weight, {MIN_EDGE:.0%} minimum edge): '
           f'<b>{current["bets"]} bets, {current["won"]:.0%} won, ROI {current["roi"]:+.1%}</b> at bet365 pre-match odds.</p>'
           if current else "")
    return f'''
<section class="card" id="track-record">
  <h2>How good is it?</h2>
  <p>Walk-forward backtest over {html.escape(bt["period"])}: the model was refitted every week using only matches played before that week.</p>
  <p>Picking the most likely result was right <b>{bt["overall"]["accuracy"]:.0%}</b> of the time across {bt["overall"]["matches"]:,} matches.
  On the big five leagues the model's log loss was <b>{vm["model_log_loss"]:.3f}</b> against <b>{vm["bet365_log_loss"]:.3f}</b> for bet365's own prices (lower is better).
  <b>The bookmaker was more accurate than the model.</b></p>
  {roi}
  <table class="small"><tr><th>Competition</th><th>Matches</th><th>Favourite won</th></tr>{rows}</table>
</section>'''


def render_html(days, generated_at=None):
    """days: list of (date, predictions). Returns a complete standalone HTML page."""
    generated_at = generated_at or datetime.now(timezone.utc)
    tabs, panels = [], []
    for n, (day, preds) in enumerate(days):
        label = day.strftime("%a %d %b")
        n_value = sum(len(p.value_bets) for p in preds)
        tabs.append(f'<button role="tab" aria-selected="{str(n == 0).lower()}" data-tab="d{n}">{label}'
                    + (f' <span class="badge">{n_value}</span>' if n_value else "") + "</button>")
        sections = []
        for league, group in groupby(_sorted(preds), key=lambda p: p.fixture.league_name):
            cards = "".join(_match_card(p) for p in group)
            sections.append(f'<section><h2>{html.escape(league)}</h2><div class="grid">{cards}</div></section>')
        body = "".join(sections) or '<p class="empty">No featured matches on this day.</p>'
        panels.append(f'<div role="tabpanel" id="d{n}"{"" if n == 0 else " hidden"}>{body}</div>')

    return TEMPLATE.format(
        generated=generated_at.strftime("%Y-%m-%d %H:%M UTC"),
        generated_iso=generated_at.isoformat(),
        tabs="".join(tabs),
        panels="".join(panels),
        backtest=_backtest_section(_load_backtest()),
        model_weight=f"{MODEL_WEIGHT:.0%}",
        market_weight=f"{1 - MODEL_WEIGHT:.0%}",
        min_edge=f"{MIN_EDGE:.0%}",
    )


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Football Predictor</title>
<meta name="description" content="Daily Champions League and top-league match predictions from a Dixon-Coles goals model.">
<style>
:root {{
  --bg: #f6f7f9; --surface: #ffffff; --text: #16191d; --muted: #5f6873; --border: #dfe3e8;
  --home: #2563eb; --draw: #8b95a1; --away: #dc2626; --accent: #0f766e; --accent-bg: #e6f4f2;
  --warn-bg: #fff7e6; --warn-border: #f1c27d;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #0f1216; --surface: #181c22; --text: #e6e9ed; --muted: #9aa4af; --border: #2a3039;
    --home: #4f8cff; --draw: #6b7581; --away: #f05252; --accent: #2dd4bf; --accent-bg: #10302c;
    --warn-bg: #2b2414; --warn-border: #6b5323;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #0f1216; --surface: #181c22; --text: #e6e9ed; --muted: #9aa4af; --border: #2a3039;
  --home: #4f8cff; --draw: #6b7581; --away: #f05252; --accent: #2dd4bf; --accent-bg: #10302c;
  --warn-bg: #2b2414; --warn-border: #6b5323;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }}
header h1 {{ margin: 0 0 4px; font-size: 1.6rem; }}
header p {{ margin: 0; color: var(--muted); }}
.warn {{ background: var(--warn-bg); border: 1px solid var(--warn-border); border-radius: 10px;
  padding: 10px 14px; margin: 16px 0; font-size: .9rem; }}
[role=tablist] {{ display: flex; gap: 6px; margin: 20px 0 8px; flex-wrap: wrap; }}
[role=tab] {{ font: inherit; padding: 6px 14px; border-radius: 999px; border: 1px solid var(--border);
  background: var(--surface); color: var(--text); cursor: pointer; }}
[role=tab][aria-selected=true] {{ background: var(--text); color: var(--bg); border-color: var(--text); }}
.badge {{ display: inline-block; min-width: 1.4em; padding: 0 5px; margin-left: 4px; border-radius: 999px;
  background: var(--accent); color: #fff; font-size: .75rem; }}
h2 {{ font-size: 1.1rem; margin: 24px 0 10px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(320px, 100%), 1fr)); gap: 12px; }}
.match, .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }}
.card {{ margin-top: 32px; }}
.card h2 {{ margin-top: 0; }}
.match.muted {{ opacity: .7; }}
time {{ color: var(--muted); font-size: .85rem; }}
.teams {{ display: grid; grid-template-columns: 1fr auto 1fr; gap: 8px; align-items: center; font-weight: 600; margin: 4px 0 10px; }}
.teams span:last-child {{ text-align: right; }}
.vs {{ color: var(--muted); font-weight: 400; font-size: .85rem; }}
.bar {{ display: flex; height: 24px; border-radius: 6px; overflow: hidden; font-size: .75rem; color: #fff; font-weight: 600; }}
.bar span {{ display: flex; align-items: center; justify-content: center; min-width: 0; overflow: hidden; white-space: nowrap; }}
.bar .h {{ background: var(--home); }} .bar .d {{ background: var(--draw); }} .bar .a {{ background: var(--away); }}
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; margin: 10px 0; }}
.stats div {{ text-align: center; }}
.stats dt {{ font-size: .72rem; color: var(--muted); }}
.stats dd {{ margin: 0; font-weight: 600; font-variant-numeric: tabular-nums; }}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: .85rem; }}
th, td {{ padding: 3px 6px; text-align: right; border-bottom: 1px solid var(--border); }}
th:first-child, td:first-child {{ text-align: left; color: var(--muted); }}
.small {{ max-width: 480px; margin-top: 12px; }}
.pick {{ margin: 8px 0 0; font-size: .9rem; }}
.pick.value {{ background: var(--accent-bg); border-radius: 8px; padding: 6px 8px; }}
.tag {{ color: var(--muted); font-size: .8rem; white-space: nowrap; }}
.note, .empty {{ color: var(--muted); font-size: .85rem; margin: 6px 0 0; }}
footer {{ margin-top: 32px; color: var(--muted); font-size: .85rem; }}
footer a {{ color: inherit; }}
</style>
</head>
<body>
<main>
<header>
  <h1>Football Predictor</h1>
  <p>Champions League, European and top-league matches · updated <time datetime="{generated_iso}" data-full>{generated}</time></p>
</header>
<p class="warn"><b>Read this first.</b> These are model estimates, not tips that make money.
In backtests the bookmaker's prices were more accurate than the model, and the value bets lost money.
Never bet more than you can afford to lose. 18+.</p>
<div role="tablist">{tabs}</div>
{panels}
{backtest}
<footer>
  <p><b>How it works:</b> a Dixon-Coles goals model rates every team's attack and defence from about four seasons of results in 28 competitions (FotMob data), weighting recent matches more.
  Team ratings are pulled towards their league's average, and European matches put all the leagues on one scale.
  "Value" means the blend of model ({model_weight}) and bookmaker ({market_weight}) gives at least {min_edge} expected profit. "Lean" is the most likely simple outcome.</p>
  <p><a href="https://github.com/Izu83/Football-Predictor">Source on GitHub</a></p>
</footer>
</main>
<script>
document.querySelectorAll('time[datetime]').forEach(function (t) {{
  var d = new Date(t.getAttribute('datetime'));
  if (isNaN(d)) return;
  t.textContent = t.hasAttribute('data-full')
    ? d.toLocaleString([], {{dateStyle: 'medium', timeStyle: 'short'}})
    : d.toLocaleTimeString([], {{hour: '2-digit', minute: '2-digit'}});
}});
document.querySelectorAll('[role=tab]').forEach(function (tab) {{
  tab.addEventListener('click', function () {{
    document.querySelectorAll('[role=tab]').forEach(function (t) {{ t.setAttribute('aria-selected', t === tab); }});
    document.querySelectorAll('[role=tabpanel]').forEach(function (p) {{ p.hidden = p.id !== tab.dataset.tab; }});
  }});
}});
</script>
</body>
</html>
"""
