"""LiveScore fixtures, used when FotMob is unavailable.

LiveScore uses its own team ids, so teams are matched to FotMob ones by name
(see `data.TeamIndex`).
"""

from datetime import date, datetime, timezone

from .fotmob import Fixture
from .net import get_json

BASE = "https://prod-public-api.livescore.com/v1/api/app/date/soccer"

# LiveScore competition names -> FotMob league ids, for the competitions we feature.
COMPETITIONS = {
    ("Champions League", ""): 42,
    ("Europa League", ""): 73,
    ("Conference League", ""): 10216,
    ("England", "Premier League"): 47,
    ("Spain", "LaLiga"): 87,
    ("Italy", "Serie A"): 55,
    ("Germany", "Bundesliga"): 54,
    ("France", "Ligue 1"): 53,
    ("Netherlands", "Eredivisie"): 57,
    ("Portugal", "Liga Portugal"): 61,
}


def _league_id(stage):
    for (cnm, snm), league_id in COMPETITIONS.items():
        if stage.get("Cnm") == cnm and (not snm or stage.get("Snm") == snm):
            return league_id
    return None


def fixtures_on(day: date, utc_offset_hours=3):
    data = get_json(f"{BASE}/{day.strftime('%Y%m%d')}/{utc_offset_hours}", {"locale": "en"}, max_age=10 * 60)
    fixtures = []
    for stage in data.get("Stages", []):
        league_name = stage.get("CompN") or f"{stage.get('Cnm')} {stage.get('Snm')}"
        for e in stage.get("Events", []):
            # Esd is the kickoff in UTC as yyyymmddHHMMSS.
            kickoff = datetime.strptime(str(e["Esd"]), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            status = e.get("Eps", "NS")
            started = status not in ("NS", "Postp.", "Canc.")
            fixtures.append(
                Fixture(
                    match_id=None,
                    kickoff=kickoff,
                    league_id=_league_id(stage),
                    league_name=league_name,
                    home_id=None,
                    home_name=e["T1"][0]["Nm"],
                    away_id=None,
                    away_name=e["T2"][0]["Nm"],
                    started=started,
                    finished=status in ("FT", "AET", "AP"),
                    score=f"{e.get('Tr1', '')}-{e.get('Tr2', '')}" if started else "",
                )
            )
    return fixtures
