"""Builds the historical results table the model is trained on."""

import difflib
import re
import unicodedata

import pandas as pd

from . import fotmob
from .config import DATA_DIR, EUROPEAN_COMPETITIONS, SEASONS_PER_LEAGUE, TRAINING_LEAGUES

RESULTS_FILE = DATA_DIR / "results.csv"


def update_results(leagues=TRAINING_LEAGUES, seasons_per_league=SEASONS_PER_LEAGUE, log=print):
    """Download every configured league season from FotMob and save them to one CSV."""
    rows = []
    for league_id, name in leagues.items():
        seasons = fotmob.league_seasons(league_id)[:seasons_per_league]
        for i, season in enumerate(seasons):
            try:
                matches = fotmob.league_matches(league_id, season, current=(i == 0))
            except Exception as exc:  # one broken season shouldn't stop the rest
                log(f"  ! {name} {season}: {exc}")
                continue
            played = sum(m["home_goals"] is not None for m in matches)
            log(f"  {name:<26} {season:<10} {played:>4} played / {len(matches)} matches")
            rows.extend(matches)

    df = pd.DataFrame(rows).drop_duplicates("match_id")
    df["european"] = df["league_id"].isin(EUROPEAN_COMPETITIONS)
    df = df.sort_values("date").reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_FILE, index=False)
    return df


def load_results():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            "No match history yet. Run `python -m football_predictor update` first."
        )
    df = pd.read_csv(RESULTS_FILE, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def played(df):
    return df.dropna(subset=["home_goals", "away_goals"]).astype(
        {"home_goals": int, "away_goals": int}
    )


def normalize_name(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    name = re.sub(r"\b(fc|cf|afc|sc|ac|as|ssc|fk|sk|nk|bk|if|ss|us|cd|ud|rc|club|calcio|1907|1909)\b", " ", name)
    name = name.replace("&", "and")
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


class TeamIndex:
    """Looks up FotMob team ids by name, for sources that only give names (LiveScore)."""

    def __init__(self, results):
        names = pd.concat(
            [results[["home_id", "home"]].set_axis(["id", "name"], axis=1),
             results[["away_id", "away"]].set_axis(["id", "name"], axis=1)]
        ).drop_duplicates()
        self.by_name = {normalize_name(n): int(i) for i, n in zip(names["id"], names["name"])}

    def find(self, name, cutoff=0.8):
        key = normalize_name(name)
        if key in self.by_name:
            return self.by_name[key]
        close = difflib.get_close_matches(key, self.by_name, n=1, cutoff=cutoff)
        return self.by_name[close[0]] if close else None
