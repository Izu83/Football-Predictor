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
    10611: "Champions League Qualification",
    10613: "Europa League Qualification",
    10615: "Conference League Qualification",
}

# Second divisions: without them newly promoted teams have almost no history.
SECOND_DIVISIONS = {
    86: "Serie B",
    140: "LaLiga2",
    146: "2. Bundesliga",
    110: "Ligue 2",
    185: "Liga Portugal 2",
    111: "Eerste Divisie",
}

# Smaller leagues whose clubs regularly reach the European competitions.
SMALLER_LEAGUES = {
    212: "Hungarian NB I",
    189: "Romanian Liga I",
    173: "Slovenian PrvaLiga",
    176: "Slovak 1. liga",
    225: "Kazakhstan Premier League",
    262: "Azerbaijan Premier League",
    118: "Armenian Premier League",
    439: "Georgian Erovnuli Liga",
    231: "Moldovan National Division",
    228: "Lithuanian A Lyga",
    226: "Latvian Virsliga",
    248: "Estonian Meistriliiga",
    51: "Veikkausliiga",
    215: "Icelandic Besta deildin",
    126: "League of Ireland",
    129: "NIFL Premiership",
    116: "Cymru Premier",
    267: "Bosnian Premier League",
    260: "Albanian Kategoria Superiore",
    249: "Macedonian Prva Liga",
    232: "Montenegrin 1. CFL",
    229: "Luxembourg National Division",
    250: "Faroese Premier League",
    263: "Belarusian Premier League",
}

DOMESTIC_LEAGUES = {
    **SECOND_DIVISIONS,
    **SMALLER_LEAGUES,
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

# Model defaults, chosen on 2024/25 with `python -m football_predictor backtest --tune`.
HALF_LIFE_DAYS = 180
TEAM_RIDGE = 2.0
LEAGUE_RIDGE = 0.5
HOME_RIDGE = 5.0

# Fallbacks only: `backtest` chooses the model/bookmaker blend and value-bet rule
# on the tuning season and saves them in reports/backtest.json.
MODEL_WEIGHT = 0.35

# Value-bet rules.
MIN_EDGE = 0.05          # expected value per unit staked
MIN_PROBABILITY = 0.25   # skip long shots, the model is least reliable there
MIN_MATCHES = 20         # per team in the last year; fewer and the rating is mostly guesswork
MAX_ODDS = 5.0
KELLY_FRACTION = 0.25
MAX_STAKE = 0.02         # fraction of bankroll per bet
