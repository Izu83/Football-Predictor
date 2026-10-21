"""Turns a score-probability matrix into betting-market probabilities and returns."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Bet:
    market: str
    selection: str
    odds: float
    model_return: float        # expected return per 1 unit staked, from the model alone
    market_return: float       # same, if the bookmaker's (de-vigged) price were exactly right
    model_win_prob: float      # probability the bet wins (excluding pushes)
    kind: str = ""             # how to settle it: "1x2", "btts", "ah" or "ou"
    side: str = ""             # home/draw/away, yes/no, over/under
    line: float = 0.0          # handicap or goal line

    def blended_ev(self, model_weight):
        return model_weight * (self.model_return - 1) + (1 - model_weight) * (self.market_return - 1)


def settle(kind, side, line, odds, home_goals, away_goals):
    """Profit per 1 unit staked once the final score is known."""
    if kind == "1x2":
        diff = home_goals - away_goals
        won = {"home": diff > 0, "draw": diff == 0, "away": diff < 0}[side]
        return odds - 1 if won else -1.0
    if kind == "btts":
        both = home_goals > 0 and away_goals > 0
        return odds - 1 if both == (side == "yes") else -1.0
    if kind == "ou":
        over = home_goals + away_goals > line
        return odds - 1 if over == (side == "over") else -1.0
    if kind == "ah":
        if abs(line * 4) % 2 == 1:  # quarter line: half the stake on each neighbouring line
            return 0.5 * (settle(kind, side, line - 0.25, odds, home_goals, away_goals)
                          + settle(kind, side, line + 0.25, odds, home_goals, away_goals))
        margin = (home_goals - away_goals if side == "home" else away_goals - home_goals) + line
        return odds - 1 if margin > 0 else 0.0 if margin == 0 else -1.0
    raise ValueError(kind)


class MatchProbabilities:
    def __init__(self, matrix):
        self.m = matrix
        n = matrix.shape[0]
        i, j = np.indices((n, n))
        self.diff = i - j
        self.total = i + j

    # outcome probabilities -------------------------------------------------
    @property
    def home(self):
        return float(self.m[self.diff > 0].sum())

    @property
    def draw(self):
        return float(self.m[self.diff == 0].sum())

    @property
    def away(self):
        return float(self.m[self.diff < 0].sum())

    @property
    def btts(self):
        return float(self.m[1:, 1:].sum())

    def over(self, line):
        return float(self.m[self.total > line].sum())

    def most_likely_score(self):
        i, j = np.unravel_index(np.argmax(self.m), self.m.shape)
        return int(i), int(j), float(self.m[i, j])

    def expected_goals(self):
        n = self.m.shape[0]
        return float(self.m.sum(1) @ np.arange(n)), float(self.m.sum(0) @ np.arange(n))

    # handicap settlement ---------------------------------------------------
    def asian_handicap_return(self, line, odds, side="home"):
        """Expected return per unit of an Asian handicap bet (quarter lines split the stake)."""
        if abs(line * 4) % 2 == 1:  # quarter line: half on each neighbouring line
            return 0.5 * (self.asian_handicap_return(line - 0.25, odds, side)
                          + self.asian_handicap_return(line + 0.25, odds, side))
        margin = (self.diff if side == "home" else -self.diff) + line
        win = self.m[margin > 0].sum()
        push = self.m[margin == 0].sum()
        return float(win * odds + push)

    def asian_handicap_win_prob(self, line, side="home"):
        margin = (self.diff if side == "home" else -self.diff) + line
        return float(self.m[margin > 0].sum())


def _fair(odds):
    """Remove the bookmaker margin proportionally; returns fair probabilities."""
    inv = np.array([1 / o for o in odds])
    return inv / inv.sum()


def _parse_line(value):
    try:
        return float(str(value).replace("+", ""))
    except (TypeError, ValueError):
        return None


def candidate_bets(probs: MatchProbabilities, odds_markets, home_name, away_name):
    """Every bet we can price from FotMob's odds, with model and market expected returns."""
    bets = []

    sels = odds_markets.get("Full Time Result")
    if sels and len(sels) == 3:
        prices = {name.lower(): o for name, _, o in sels}
        if {"1", "x", "2"} <= prices.keys():
            odds = [prices["1"], prices["x"], prices["2"]]
            fair = _fair(odds)
            for label, side, p, o, q in zip((home_name, "Draw", away_name), ("home", "draw", "away"),
                                            (probs.home, probs.draw, probs.away), odds, fair):
                bets.append(Bet("1X2", label, o, p * o, q * o, p, "1x2", side))

    sels = odds_markets.get("Both Teams to Score")
    if sels and len(sels) == 2:
        prices = {name.lower(): o for name, _, o in sels}
        if {"yes", "no"} <= prices.keys():
            fair = _fair([prices["yes"], prices["no"]])
            for label, side, p, o, q in zip(("BTTS Yes", "BTTS No"), ("yes", "no"), (probs.btts, 1 - probs.btts),
                                            (prices["yes"], prices["no"]), fair):
                bets.append(Bet("BTTS", label, o, p * o, q * o, p, "btts", side))

    sels = odds_markets.get("Draw No Bet")
    if sels and len(sels) == 2:
        (_, _, o_home), (_, _, o_away) = sels
        fair = _fair([o_home, o_away])
        for label, side, o, q in ((home_name, "home", o_home, fair[0]), (away_name, "away", o_away, fair[1])):
            bets.append(Bet("Draw No Bet", label, o,
                            probs.asian_handicap_return(0, o, side), q * o,
                            probs.asian_handicap_win_prob(0, side), "ah", side, 0.0))

    sels = odds_markets.get("Asian Handicap")
    if sels and len(sels) == 2:
        (l_home, _, o_home), (l_away, _, o_away) = sels
        line = _parse_line(l_home)
        if line is not None:
            fair = _fair([o_home, o_away])
            for label, side, ln, o, q in (
                (f"{home_name} {l_home}", "home", line, o_home, fair[0]),
                (f"{away_name} {l_away}", "away", -line, o_away, fair[1]),
            ):
                bets.append(Bet("Asian Handicap", label, o,
                                probs.asian_handicap_return(ln, o, side), q * o,
                                probs.asian_handicap_win_prob(ln, side), "ah", side, ln))

    # Over/under is not always offered; handle the common header names.
    for header in ("Match Goals", "Total Goals", "Goals Over/Under", "Over/Under"):
        sels = odds_markets.get(header)
        if not sels or len(sels) != 2:
            continue
        line = _parse_line(sels[0][1])
        if line is None or line != int(line) + 0.5:
            continue  # only half-goal lines (no pushes)
        names = [s[0].lower() for s in sels]
        if not (names[0].startswith("o") and names[1].startswith("u")):
            continue
        o_over, o_under = sels[0][2], sels[1][2]
        fair = _fair([o_over, o_under])
        p_over = probs.over(line)
        bets.append(Bet("Over/Under", f"Over {line}", o_over, p_over * o_over, fair[0] * o_over, p_over, "ou", "over", line))
        bets.append(Bet("Over/Under", f"Under {line}", o_under, (1 - p_over) * o_under, fair[1] * o_under, 1 - p_over,
                        "ou", "under", line))
        break

    return bets
