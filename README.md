# Football Predictor

Predicts Champions League, Europa League, Conference League and top-league matches. It pulls the day's fixtures and bookmaker odds from FotMob and suggests what to bet on. It also publishes a daily page to GitHub Pages.

> **Read this before betting.** In a full-season backtest (2025/26), bet365's own prices predicted results better than this model, and every value-bet setting lost money. Use the output as a second opinion, not as a way to make money. Never stake more than you can afford to lose. 18+.

## What it does

1. **Downloads history.** Pulls about four seasons of results from 28 competitions on FotMob (about 25,000 matches): the Champions League, Europa League and Conference League, plus 25 European leagues.
2. **Rates every team.** A [Dixon-Coles](https://en.wikipedia.org/wiki/Dixon%E2%80%93Coles_model) goals model gives each team an attack and a defence rating:
   - Recent matches count more (half-life of 240 days).
   - Each team's rating is pulled toward its league's average, so newly promoted teams start from a sensible guess.
   - European matches link the leagues, so a Czech side can be compared with an English one.
3. **Predicts each match.** It works out the chance of every scoreline. From that it gives the 1X2 probabilities, expected goals, the likeliest score, over/under 2.5 and both teams to score.
4. **Compares with the bookmaker.** It fetches the bookmaker's odds from FotMob (bet365 by default) for 1X2, Draw No Bet, Asian Handicap and BTTS. It blends 35% model with 65% bookmaker probability, and flags a **value bet** when that blend still expects at least 5% profit. Stakes use quarter-Kelly, capped at 2% of bankroll.
5. **Suggests a lean.** For every match it also names the most likely simple outcome, such as "Inter to win" or "Over 2.5 goals".

Value bets are skipped when either team has played fewer than 20 matches in the past year. That usually means the team's league isn't in the dataset, such as Slovenia or Andorra, or it was just promoted.

## Backtest results (2025/26 season)

The model was refitted every week using only matches played before that week, then scored on the matches that followed. Run it yourself with `python -m football_predictor backtest`; the summary is saved to [`reports/backtest.json`](reports/backtest.json).

| | Matches | Log loss (lower is better) | Favourite won |
|---|---:|---:|---:|
| Model, all 28 competitions | 7,482 | 1.003 | 51% |
| Always predicting the average result | 7,482 | 1.074 | – |
| Model, big five leagues | 1,745 | 0.998 | – |
| bet365 prices (margin removed), big five | 1,745 | **0.978** | – |
| Pinnacle closing prices, big five | 891 | **0.962** | – |

The model clearly beats guessing, but the bookmakers are more accurate. Value-bet simulations at bet365 pre-match odds over the same season (flat stakes):

| Model weight | Min edge | Bets | ROI |
|---:|---:|---:|---:|
| 35% (default) | 5% | 308 | −6.3% |
| 35% | 3% | 587 | −9.2% |
| 50% | 10% | 257 | −7.6% |
| 100% | 5% | 2,240 | −9.9% |

## Usage

Needs Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

Download the match history (about 2 minutes the first time; finished seasons are cached in `data/`):

```bash
python -m football_predictor update
```

Predict today's big matches and suggest bets:

```bash
python -m football_predictor predict
```

Useful options: `--date 2026-10-22`, `--days 3`, `--all` (every competition FotMob lists, not just the featured ones), `--no-odds`, and `--html index.html` to also write the web page.

Other commands:

```bash
python -m football_predictor ratings --top 30
```

```bash
python -m football_predictor backtest --tune
```

Settings (featured competitions, model parameters, value-bet thresholds, bookmaker country) are in [`football_predictor/config.py`](football_predictor/config.py). The odds come from the bookmaker FotMob shows for `ODDS_COUNTRY` (`BGR` gives bet365; some countries, like the USA, get no odds).

## GitHub Pages

`index.html` is a static page with predictions for today and the next two days. The workflow in [`.github/workflows/update-predictions.yml`](.github/workflows/update-predictions.yml) rebuilds it every day at 06:00 UTC, commits it and publishes it to GitHub Pages.

To turn it on once:

1. Go to the repository's **Settings → Pages** and set **Source** to **GitHub Actions**.
2. Go to the **Actions** tab, open **Update predictions** and click **Run workflow**.

The page will then be at `https://izu83.github.io/Football-Predictor/`. The workflow commits to `main` every day, so run `git pull` before you push changes.

## Data sources

- [FotMob](https://www.fotmob.com): fixtures, results and odds, through its unofficial web API. It can change without notice.
- [LiveScore](https://www.livescore.com): fallback for fixtures when FotMob is down. Its teams are matched to FotMob teams by name.
- [football-data.co.uk](https://www.football-data.co.uk): historical bet365 and Pinnacle odds, used only by the backtest.

## Project structure

```
football_predictor/
  config.py      competitions, model and betting settings
  fotmob.py      FotMob client (fixtures, league seasons, odds)
  livescore.py   LiveScore fallback for fixtures
  data.py        builds and loads the results history
  model.py       hierarchical Dixon-Coles model
  markets.py     scoreline probabilities -> market probabilities, odds parsing
  betting.py     value-bet selection, Kelly staking, leans
  backtest.py    walk-forward evaluation against results and bookmaker odds
  report.py      console output and the HTML page
  cli.py         command line
tests/           unit tests (pytest)
reports/         saved backtest summary
index.html       the GitHub Pages site
```

## License

No license has been chosen yet.
