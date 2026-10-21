"""Project-wide settings: which competitions to use, where to cache data, model defaults."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = ROOT / "reports"

# Country code FotMob uses to pick a bookmaker for odds (BGR -> bet365).
# Some countries (e.g. USA) get no odds at all.
ODDS_COUNTRY = "BGR"
# Timezone FotMob uses to decide which matches belong to "today".
TIMEZONE = "Europe/Sofia"

# FotMob league ids. European competitions link the domestic leagues together,
# which is what lets the model compare e.g. a Czech team with an English one.
EUROPEAN_COMPETITIONS = {
    42: "Champions League",
    73: "Europa League",
    10216: "Conference League",
}

DOMESTIC_LEAGUES = {
    47: "Premier League",
    48: "Championship",
    87: "LaLiga",
    55: "Serie A",
    54: "Bundesliga",
    53: "Ligue 1",
    57: "Eredivisie",
    61: "Liga Portugal",
    40: "Belgian Pro League",
    71: "Super Lig",
    64: "Scottish Premiership",
    38: "Austrian Bundesliga",
    69: "Swiss Super League",
    135: "Greek Super League",
    46: "Danish Superliga",
    122: "Czech First League",
    59: "Eliteserien",
    67: "Allsvenskan",
    252: "Croatian HNL",
    182: "Serbian SuperLiga",
    441: "Ukrainian Premier League",
    136: "Cypriot First Division",
    196: "Ekstraklasa",
    270: "Bulgarian First League",
    127: "Israeli Premier League",
}

TRAINING_LEAGUES = {**EUROPEAN_COMPETITIONS, **DOMESTIC_LEAGUES}

# How many seasons of history to pull per league (most recent first).
SEASONS_PER_LEAGUE = 4

# Competitions shown by `predict` by default: the "big matches" of the day.
FEATURED_COMPETITIONS = {
    **EUROPEAN_COMPETITIONS,
    47: "Premier League",
    87: "LaLiga",
    55: "Serie A",
    54: "Bundesliga",
    53: "Ligue 1",
    57: "Eredivisie",
    61: "Liga Portugal",
    132: "FA Cup",
    133: "EFL Cup",
    138: "Copa del Rey",
    141: "Coppa Italia",
    209: "DFB Pokal",
    134: "Coupe de France",
}

# Model defaults (tuned with `python -m football_predictor backtest`).
HALF_LIFE_DAYS = 240
TEAM_RIDGE = 3.0
LEAGUE_RIDGE = 0.5

# Weight of the model vs the de-vigged bookmaker price when both are available.
# In the 2025/26 backtest bet365's prices were more accurate than the model on
# their own, so the model only nudges the market price.
MODEL_WEIGHT = 0.35

# Value-bet rules.
MIN_EDGE = 0.05          # expected value per unit staked
MIN_PROBABILITY = 0.25   # skip long shots, the model is least reliable there
MIN_MATCHES = 20         # per team in the last year; fewer and the rating is mostly guesswork
MAX_ODDS = 5.0
KELLY_FRACTION = 0.25
MAX_STAKE = 0.02         # fraction of bankroll per bet
