"""Command line: `python -m football_predictor <command>`."""

import argparse
from datetime import date, datetime, timedelta

from . import backtest, data, fotmob, history, livescore
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
    if args.track:
        print("Scoring finished matches in the track record...")
        history.settle_finished()
    model = DixonColesModel().fit(data.load_results())
    start = date.fromisoformat(args.date) if args.date else date.today()
    days = []
    for n in range(args.days):
        day = start + timedelta(days=n)
        fixtures = get_fixtures(day, args.all)
        preds = predict_fixtures(model, fixtures, with_odds=not args.no_odds)
        days.append((day, preds))
        if args.track:
            history.record(preds)
        print(f"\n##### {day:%A %d %B %Y} — {len(preds)} matches #####")
        print_predictions(preds)
    if args.html:
        path = ROOT / args.html
        path.write_text(render_html(days, track_record=history.summary()), encoding="utf-8")
        print(f"\nWrote {path}")


def cmd_backtest(args):
    results = data.load_results()
    params = {}
    if args.tune:
        print("Tuning model parameters on 2024/25...")
        grid = [dict(half_life_days=h, team_ridge=r) for h in (180, 240, 365) for r in (2.0, 3.0, 5.0)]
        params = backtest.tune_model(results, grid)
    backtest.run(results, model_kwargs=params)


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
    p.add_argument("--track", action="store_true",
                   help="save predictions to history/predictions.csv and score finished ones (used by the daily workflow)")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("backtest", help="tune on 2024/25, test on 2025/26 against real results and odds")
    p.add_argument("--tune", action="store_true", help="also search model parameters (on 2024/25 only)")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("ratings", help="show the strongest teams")
    p.add_argument("--top", type=int, default=30)
    p.set_defaults(func=cmd_ratings)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
