"""Hierarchical Dixon-Coles goals model.

Each team has an attack and a defence rating. A team's rating is its league's
rating plus a team-specific offset; ridge penalties pull offsets towards the
league average, so teams with little data (e.g. newly promoted sides) start
from a sensible prior instead of from zero. European matches put every league
on one common scale.

    log(home xG) = mu + home_adv[competition] + attack[home] - defence[away]
    log(away xG) = mu                         + attack[away] - defence[home]

Goals are Poisson with the Dixon-Coles correction for low scores (0-0, 1-0,
0-1, 1-1). Older matches count less (exponential decay with a half-life).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from .config import (
    DOMESTIC_LEAGUES,
    HALF_LIFE_DAYS,
    HOME_RIDGE,
    LEAGUE_RIDGE,
    TEAM_RIDGE,
)

MAX_GOALS = 10
OTHER_LEAGUE = -1  # teams we only see in European competitions
EUROPE = -2        # home-advantage key shared by all European matches


@dataclass
class TeamRating:
    team_id: int
    name: str
    league_id: int
    attack: float
    defence: float
    matches_last_year: int


class DixonColesModel:
    def __init__(self, half_life_days=HALF_LIFE_DAYS, team_ridge=TEAM_RIDGE, league_ridge=LEAGUE_RIDGE,
                 home_ridge=HOME_RIDGE):
        self.half_life_days = half_life_days
        self.team_ridge = team_ridge
        self.league_ridge = league_ridge
        self.home_ridge = home_ridge

    # ------------------------------------------------------------------ fit

    def fit(self, results: pd.DataFrame, as_of=None):
        """Fit on played matches strictly before `as_of` (default: all of them)."""
        df = results.dropna(subset=["home_goals", "away_goals"])
        if as_of is not None:
            df = df[df["date"] < as_of]
        as_of = as_of if as_of is not None else df["date"].max()
        age_days = (as_of - df["date"]).dt.total_seconds().to_numpy() / 86400
        # Ignore very old matches entirely: they carry almost no weight anyway.
        keep = age_days < 6 * self.half_life_days
        df, age_days = df[keep], age_days[keep]
        w = 0.5 ** (age_days / self.half_life_days)

        self.as_of = as_of
        self._index_teams(df, as_of)

        hi = df["home_id"].map(self.team_idx).to_numpy()
        ai = df["away_id"].map(self.team_idx).to_numpy()
        hl = self.team_league[hi]
        al = self.team_league[ai]
        # Each domestic competition has its own home advantage; European ones share one.
        comp_keys = np.where(df["european"].to_numpy().astype(bool), EUROPE, df["league_id"].to_numpy())
        self.comp_ids = sorted(set(comp_keys.tolist()))
        comp_idx = {c: i for i, c in enumerate(self.comp_ids)}
        ci = np.array([comp_idx[c] for c in comp_keys])
        yh = df["home_goals"].to_numpy(float)
        ya = df["away_goals"].to_numpy(float)
        nT, nL, nC = len(self.team_ids), len(self.league_ids), len(self.comp_ids)
        o = 2 + nC  # where the league parameters start

        def unpack(theta):
            mu, h0 = theta[:2]
            hc = theta[2:o]
            A = theta[o:o + nL]
            D = theta[o + nL:o + 2 * nL]
            a = theta[o + 2 * nL:o + 2 * nL + nT]
            d = theta[o + 2 * nL + nT:]
            return mu, h0, hc, A, D, a, d

        def loss_grad(theta):
            mu, h0, hc, A, D, a, d = unpack(theta)
            log_lh = mu + h0 + hc[ci] + A[hl] + a[hi] - D[al] - d[ai]
            log_la = mu + A[al] + a[ai] - D[hl] - d[hi]
            lh, la = np.exp(log_lh), np.exp(log_la)
            nll = np.sum(w * (lh - yh * log_lh + la - ya * log_la))
            rh, ra = w * (lh - yh), w * (la - ya)

            g = np.zeros_like(theta)
            g[0] = rh.sum() + ra.sum()
            g[1] = rh.sum()
            g[2:o] = np.bincount(ci, rh, nC)
            g[o:o + nL] = np.bincount(hl, rh, nL) + np.bincount(al, ra, nL)
            g[o + nL:o + 2 * nL] = -np.bincount(al, rh, nL) - np.bincount(hl, ra, nL)
            g[o + 2 * nL:o + 2 * nL + nT] = np.bincount(hi, rh, nT) + np.bincount(ai, ra, nT)
            g[o + 2 * nL + nT:] = -np.bincount(ai, rh, nT) - np.bincount(hi, ra, nT)

            home_part = theta[2:o]
            league_part = theta[o:o + 2 * nL]
            team_part = theta[o + 2 * nL:]
            nll += (self.home_ridge * home_part @ home_part
                    + self.league_ridge * league_part @ league_part
                    + self.team_ridge * team_part @ team_part)
            g[2:o] += 2 * self.home_ridge * home_part
            g[o:o + 2 * nL] += 2 * self.league_ridge * league_part
            g[o + 2 * nL:] += 2 * self.team_ridge * team_part
            return nll, g

        theta0 = np.zeros(o + 2 * nL + 2 * nT)
        theta0[0] = np.log(max(np.average(ya, weights=w), 0.1))
        theta0[1] = 0.2
        res = minimize(loss_grad, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 3000})
        mu, h0, hc, A, D, a, d = unpack(res.x)
        self.mu = mu
        self.home_adv_default = h0
        self.home_adv = {c: h0 + h for c, h in zip(self.comp_ids, hc)}
        self.attack = A[self.team_league] + a
        self.defence = D[self.team_league] + d
        self.league_attack = dict(zip(self.league_ids, A))
        self.league_defence = dict(zip(self.league_ids, D))

        # Dixon-Coles low-score correlation, fitted on the same weighted data.
        lh = np.exp(mu + h0 + hc[ci] + self.attack[hi] - self.defence[ai])
        la = np.exp(mu + self.attack[ai] - self.defence[hi])
        low = (yh <= 1) & (ya <= 1)

        def rho_nll(rho):
            tau = _tau(yh[low], ya[low], lh[low], la[low], rho)
            return -np.sum(w[low] * np.log(np.clip(tau, 1e-10, None)))

        self.rho = minimize_scalar(rho_nll, bounds=(-0.2, 0.2), method="bounded").x
        return self

    def _index_teams(self, df, as_of):
        teams = pd.concat(
            [df[["date", "home_id", "home", "league_id", "european"]].set_axis(["date", "id", "name", "league_id", "european"], axis=1),
             df[["date", "away_id", "away", "league_id", "european"]].set_axis(["date", "id", "name", "league_id", "european"], axis=1)]
        ).sort_values("date")
        self.team_ids = np.array(sorted(teams["id"].unique()))
        self.team_idx = {t: i for i, t in enumerate(self.team_ids)}
        self.team_names = teams.groupby("id")["name"].last().to_dict()

        # A team belongs to the domestic league it played in most recently.
        domestic = teams[~teams["european"] & teams["league_id"].isin(DOMESTIC_LEAGUES)]
        current_league = domestic.groupby("id")["league_id"].last()
        team_league_ids = [int(current_league.get(t, OTHER_LEAGUE)) for t in self.team_ids]
        self.league_ids = sorted(set(team_league_ids))
        league_idx = {l: i for i, l in enumerate(self.league_ids)}
        self.team_league = np.array([league_idx[l] for l in team_league_ids])
        self.team_league_id = dict(zip(self.team_ids, team_league_ids))

        recent = teams[teams["date"] > as_of - pd.Timedelta(days=365)]
        self.matches_last_year = recent.groupby("id").size().to_dict()

    # -------------------------------------------------------------- predict

    def knows(self, team_id):
        return team_id in self.team_idx

    def home_advantage(self, league_id=None, european=False):
        return self.home_adv.get(EUROPE if european else league_id, self.home_adv_default)

    def expected_goals(self, home_id, away_id, european=False, neutral=False, league_id=None):
        h, a = self.team_idx[home_id], self.team_idx[away_id]
        home_adv = 0.0 if neutral else self.home_advantage(league_id, european)
        lh = np.exp(self.mu + home_adv + self.attack[h] - self.defence[a])
        la = np.exp(self.mu + self.attack[a] - self.defence[h])
        return lh, la

    def score_matrix(self, home_id, away_id, european=False, neutral=False, league_id=None):
        """P(home scores i, away scores j) for i, j in 0..MAX_GOALS."""
        lh, la = self.expected_goals(home_id, away_id, european, neutral, league_id)
        return score_matrix(lh, la, self.rho)

    def ratings(self):
        rows = [
            TeamRating(
                team_id=int(t),
                name=self.team_names[t],
                league_id=self.team_league_id[t],
                attack=float(self.attack[i]),
                defence=float(self.defence[i]),
                matches_last_year=int(self.matches_last_year.get(t, 0)),
            )
            for t, i in self.team_idx.items()
        ]
        df = pd.DataFrame(rows)
        df["strength"] = df["attack"] + df["defence"]
        return df.sort_values("strength", ascending=False).reset_index(drop=True)


def _tau(x, y, lh, la, rho):
    tau = np.ones_like(lh, dtype=float)
    tau = np.where((x == 0) & (y == 0), 1 - lh * la * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1 + lh * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + la * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return tau


def score_matrix(lh, la, rho=0.0):
    goals = np.arange(MAX_GOALS + 1)
    m = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
    m[0, 0] *= 1 - lh * la * rho
    m[0, 1] *= 1 + lh * rho
    m[1, 0] *= 1 + la * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()
