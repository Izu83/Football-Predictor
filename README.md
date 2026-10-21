# Football Predictor

Daily predictions for Champions League, Europa League, Conference League and top-league matches. Each one comes with a confidence level that has been measured on real results, plus a public track record of every prediction it has published.

**Website:** https://izu83.github.io/Football-Predictor/ (updated twice a day)

> **Can you trust it?** Partly, and the page says exactly how far. Its percentages are honest: when it says 78%, that happened about 78% of the time on a season it had never seen. But it does **not** beat the bookmakers. Their prices are more accurate than the model on its own, and betting on the model's disagreements with them lost money. So the page recommends no bets. Winning often is not the same as profit: a 78% pick at odds of 1.25 still loses money over time. 18+, bet responsibly.

## What you get for each match

- **Chance** of a home win, draw or away win. When bet365 odds are available, this is their price with the margin removed (testing showed that is the most accurate number). Otherwise it comes from the model.
- **Pick**: the most likely of eight simple outcomes (win, win or draw, over/under 2.5 goals, both teams to score). Outcomes above 85% are skipped because they pay almost nothing.
- **Confidence**: High, Medium or Low, each with its measured hit rate from testing.
- **Details**: the model's own probabilities next to the bookmaker's, expected goals, the most likely scorelines, over 2.5 and both teams to score.
- A warning when a team has too little recent data to rate reliably.

## How it was tested

`python -m football_predictor backtest` runs a walk-forward test. Every week the model is refitted using only earlier matches, then it predicts that week's games. **All settings are chosen on 2024/25 and then measured once on 2025/26**, a season that played no part in choosing them. Bookmaker odds come from football-data.co.uk. The summary is saved to [`reports/backtest.json`](reports/backtest.json), and the website shows it.

Test season 2025/26:

| | Matches | Log loss (lower is better) |
|---|---:|---:|
| Model, 61 competitions | 13,923 | 1.006 |
| Always guessing the average result | 13,923 | 1.074 |
| Model, big five leagues | 1,752 | 0.997 |
| bet365 prices (margin removed), big five | 1,752 | **0.979** |
| Pinnacle closing prices, big five | 898 | **0.964** |

How often the picks won:

| Confidence | Picks | Stated chance | Actually won |
|---|---:|---:|---:|
| High (with bookmaker odds) | 1,019 | 78% | 79% |
| Medium (with bookmaker odds) | 733 | 68% | 72% |
| High (model only) | 7,435 | 78% | 77% |
| Medium (model only) | 6,477 | 68% | 68% |

Value betting: the best rule found on 2024/25 lost 8.8% there. On 2025/26 it lost **9.3%** over 2,811 bets, and the 90% range was −12.6% to −5.9%. Because even the optimistic end is a loss, value bets switch off automatically. They come back only if a future backtest shows a profitable range.

## Live track record

The daily workflow saves every prediction to [`history/predictions.csv`](history/predictions.csv) before kickoff and scores it after the final whistle. The website's **Track record** tab shows the results: hit rate by confidence level, accuracy against the bookmaker, and the latest results.

## How the model works

- **Data:** about four seasons from 61 FotMob competitions (about 50,000 matches). That covers the UEFA competitions and their qualifiers, top divisions, second divisions (so promoted teams have history) and smaller European leagues whose clubs reach UEFA competitions.
- **Model:** a [Dixon-Coles](https://en.wikipedia.org/wiki/Dixon%E2%80%93Coles_model) goals model:
  - each team has an attack and a defence rating, with recent matches counting more (half-life of 180 days);
  - ratings are pulled toward the team's league average, and European games put all the leagues on one scale;
  - each competition has its own home advantage.
- **Markets:** every possible scoreline gets a probability, and all the markets are worked out from those.

## Usage

Needs Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

Download the match history (a few minutes the first time; finished seasons are cached in `data/`):

```bash
python -m football_predictor update
```

Predict today's big matches:

```bash
python -m football_predictor predict
```

Options: `--date 2026-10-22`, `--days 3`, `--all` (every competition, not just the featured ones), `--no-odds`, `--html index.html` (also write the website), `--track` (save to and score the track record; the workflow uses this).

Other commands:

```bash
python -m football_predictor backtest --tune
```

```bash
python -m football_predictor ratings --top 30
```

```bash
python -m pytest
```

Settings (featured competitions, model defaults, bookmaker country) are in [`football_predictor/config.py`](football_predictor/config.py). Odds come from the bookmaker FotMob shows for `ODDS_COUNTRY` (`BGR` gives bet365).

## GitHub Pages

[`.github/workflows/update-predictions.yml`](.github/workflows/update-predictions.yml) runs at 06:00 and 15:00 UTC and on every code change. It downloads fresh data, scores finished predictions, predicts the next three days, commits `index.html` and the track record, and publishes the site.

To set it up once: in **Settings → Pages**, set **Source** to **GitHub Actions**, then use **Actions → Update predictions → Run workflow**. The workflow commits to `main`, so run `git pull` before pushing your own changes.

## Data sources

- [FotMob](https://www.fotmob.com): fixtures, results, odds and team logos, through its unofficial web API. It can change without notice.
- [LiveScore](https://www.livescore.com): fallback for fixtures when FotMob is down.
- [football-data.co.uk](https://www.football-data.co.uk): historical bet365 and Pinnacle odds, used only for testing.

## Project structure

```
football_predictor/
  config.py      competitions, model defaults, bookmaker country
  fotmob.py      FotMob client (fixtures, seasons, odds, results)
  livescore.py   LiveScore fallback for fixtures
  data.py        builds and loads the results history
  model.py       hierarchical Dixon-Coles model
  markets.py     scoreline probabilities -> markets, odds parsing, bet settlement
  picks.py       blending with the bookmaker, picks and confidence levels
  betting.py     predicts fixtures
  history.py     live track record
  backtest.py    tune-on-one-season, test-on-the-next evaluation
  report.py      console output and the website data
  page.html      the website template
  cli.py         command line
tests/           unit tests (pytest)
reports/         saved backtest summary
history/         live predictions and their results
index.html       the generated website
```

## License

No license has been chosen yet.
