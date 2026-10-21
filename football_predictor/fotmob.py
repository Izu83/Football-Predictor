"""FotMob API client: fixtures by date, full league seasons, and bookmaker odds."""

from dataclasses import dataclass
from datetime import date, datetime

from .config import ODDS_COUNTRY, TIMEZONE
from .net import get_json

BASE = "https://www.fotmob.com/api/data"
HOUR = 3600


@dataclass
class Fixture:
    match_id: int | None
    kickoff: datetime  # UTC
    league_id: int | None
    league_name: str
    home_id: int | None
    home_name: str
    away_id: int | None
    away_name: str
    started: bool = False
    finished: bool = False
    score: str = ""


def _parse_utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def fixtures_on(day: date, timezone=TIMEZONE):
    """All matches FotMob lists for a day."""
    data = get_json(
        f"{BASE}/matches",
        {"date": day.strftime("%Y%m%d"), "timezone": timezone},
        max_age=10 * 60,
    )
    fixtures = []
    for league in data.get("leagues", []):
        for m in league.get("matches", []):
            status = m.get("status", {})
            if status.get("cancelled"):
                continue
            score = ""
            if status.get("started"):
                score = f"{m['home'].get('score', 0)}-{m['away'].get('score', 0)}"
            fixtures.append(
                Fixture(
                    match_id=m["id"],
                    kickoff=_parse_utc(status["utcTime"]),
                    league_id=league.get("primaryId"),
                    league_name=league.get("name", ""),
                    home_id=m["home"]["id"],
                    home_name=m["home"]["name"],
                    away_id=m["away"]["id"],
                    away_name=m["away"]["name"],
                    started=bool(status.get("started")),
                    finished=bool(status.get("finished")),
                    score=score,
                )
            )
    return fixtures


def league_seasons(league_id):
    """Season names FotMob has for a league, most recent first."""
    data = get_json(f"{BASE}/leagues", {"id": league_id}, max_age=24 * HOUR)
    return data.get("allAvailableSeasons", [])


def league_matches(league_id, season, current=False):
    """All matches of one league season as plain dicts."""
    data = get_json(
        f"{BASE}/leagues",
        {"id": league_id, "season": season},
        # A finished season never changes; the current one is refreshed twice a day.
        max_age=12 * HOUR if current else float("inf"),
    )
    league_name = data.get("details", {}).get("name", str(league_id))
    rows = []
    for m in data.get("fixtures", {}).get("allMatches", []):
        status = m.get("status", {})
        row = {
            "match_id": int(m["id"]),
            "date": _parse_utc(status["utcTime"]),
            "league_id": league_id,
            "league": league_name,
            "season": season,
            "home_id": int(m["home"]["id"]),
            "home": m["home"]["name"],
            "away_id": int(m["away"]["id"]),
            "away": m["away"]["name"],
            "home_goals": None,
            "away_goals": None,
        }
        score = status.get("scoreStr", "")
        if status.get("finished") and not status.get("cancelled") and " - " in score:
            # Awarded matches (walkovers) say nothing about team strength.
            if not status.get("awarded"):
                # Knockout ties report the 90/120-minute score; penalties are not counted.
                h, a = score.split(" - ")
                row["home_goals"], row["away_goals"] = int(h), int(a)
        rows.append(row)
    return rows


def match_odds(match_id, country=ODDS_COUNTRY):
    """Bookmaker odds for a match, as {market header: [(selection, line, decimal odds)]}."""
    try:
        data = get_json(
            f"{BASE}/matchOdds", {"matchId": match_id, "ccode3": country}, max_age=15 * 60
        )
    except Exception:
        return {}, None
    if not data:
        return {}, None
    bookmaker = data.get("persistentKey", "").split("_")[0] or None
    markets = {}
    for category in data.get("odds", {}).get("oddsTabMarkets", []):
        for market in category.get("markets", []):
            sels = []
            for s in market.get("selections", []):
                try:
                    sels.append((s["name"], s.get("value"), float(s["oddsDecimal"])))
                except (KeyError, TypeError, ValueError):
                    continue
            markets[market["header"]] = sels
    return markets, bookmaker
