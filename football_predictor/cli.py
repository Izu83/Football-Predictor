"""Command line: `python -m football_predictor <command>`."""

import argparse
from datetime import date, datetime, timedelta

from . import backtest, data, fotmob, livescore
from .betting import predict_fixtures
from .config import FEATURED_COMPETITIONS, ROOT
from .model import DixonColesModel
from .report import print_predictions, render_html


def get_fixtures(day, all_matches, log=print):
    try:
        fixtures = fotmob.fixtures_on(day)
    except Exception as exc:
        log(f"FotMob unavailable ({exc}); falling back to LiveScore.")
        index = data.TeamIndex(data.load_results())
        fixtures = livescore.fixtures_on(day)
        for fx in fixtures:
            fx.home_id = index.find(fx.home_name)
            fx.away_id = index.find(fx.away_name)
    if not all_matches:
        fixtures = [f for f in fixtures if f.league_id in FEATURED_COMPETITIONS]
    return fixtures


def cmd_update(args):
    print("Downloading match history from FotMob...")
    df = data.update_results()
    print(f"Saved {len(data.played(df)):,} played matches to {data.RESULTS_FILE}")


def cmd_predict(args):
    model = DixonColesModel().fit(data.load_results())
    start = date.fromisoformat(args.date) if args.date else date.today()
    days = []
    for n in range(args.days):
        day = start + timedelta(days=n)
        fixtures = get_fixtures(day, args.all)
        preds = predict_fixtures(model, fixtures, with_odds=not args.no_odds)
        days.append((day, preds))
        print(f"\n##### {day:%A %d %B %Y} — {len(preds)} matches #####")
        print_predictions(preds)
    if args.html:
        path = ROOT / args.html
        path.write_text(render_html(days), encoding="utf-8")
        print(f"\nWrote {path}")


def cmd_backtest(args):
    results = data.load_results()
    grid = None
    if args.tune:
        grid = [dict(half_life_days=h, team_ridge=r) for h in (120, 240, 365) for r in (1.0, 3.0, 6.0)]
    backtest.run(results, start=args.start, end=args.end, grid=grid)


def cmd_ratings(args):
    model = DixonColesModel().fit(data.load_results())
    r = model.ratings().head(args.top)
    for n, row in enumerate(r.itertuples(), 1):
        print(f"{n:>3}. {row.name:<28} attack {row.attack:+.2f}  defence {row.defence:+.2f}  "
              f"strength {row.strength:+.2f}  ({row.matches_last_year} matches last year)")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="football_predictor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("update", help="download match history from FotMob").set_defaults(func=cmd_update)

    p = sub.add_parser("predict", help="predict matches and suggest bets")
    p.add_argument("--date", help="YYYY-MM-DD (default: today)")
    p.add_argument("--days", type=int, default=1, help="number of days from --date")
    p.add_argument("--all", action="store_true", help="include every competition, not just the featured ones")
    p.add_argument("--no-odds", action="store_true", help="skip fetching bookmaker odds")
    p.add_argument("--html", metavar="FILE", help="also write a static HTML page, e.g. index.html")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("backtest", help="walk-forward test against real results and odds")
    p.add_argument("--start", default="2025-08-01")
    p.add_argument("--end", default="2026-06-01")
    p.add_argument("--tune", action="store_true", help="also search model parameters")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("ratings", help="show the strongest teams")
    p.add_argument("--top", type=int, default=30)
    p.set_defaults(func=cmd_ratings)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
