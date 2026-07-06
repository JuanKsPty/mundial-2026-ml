# Builds the model features for an arbitrary upcoming match, mirroring the
# training-time semantics of match_features.py:
#   * elo_diff uses the real Elo table from team_strength_2026.csv.
#   * head-to-head is built from the actual historical results.
#   * h2h defaults for unseen pairs are zeros (same as training time).

"""Pre-match feature construction for upcoming (not yet played) matches."""

from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import MODERN_TEAMS, PROCESSED_DIR, RAW_DIR, WC2026_ALL_TEAMS
from src.data.historical_data import load_and_normalize_results
from src.utils.labels import normalize_team_name

CURRENT_YEAR = 2026

# Map any alias back to the official WC 2026 draw name
_WC_TEAM_LOOKUP = {normalize_team_name(team): team for team in WC2026_ALL_TEAMS}


def to_wc_team(name: str) -> str:
    return _WC_TEAM_LOOKUP.get(normalize_team_name(name), name)


class LiveState:
    """Current strength/form/h2h state of every national team, built from the
    processed tables. WC2026 matches already played DO update Elo and form
    here (they are in results.csv), which is legitimate for prediction - the
    training cutoff only applies to model fitting."""

    def __init__(self):
        self.fifa_points: dict[str, float] = {}
        self.elo: dict[str, float] = {}
        self.last5_form: dict[str, float] = {}
        self.penalty_win_rate: dict[str, float] = {}
        self.h2h: dict[tuple, dict] = {}
        self.achievements: pd.DataFrame | None = None
        self._load_data()

    def _load_data(self):
        strength_path = PROCESSED_DIR / "team_strength_2026.csv"
        if strength_path.exists():
            strength = pd.read_csv(strength_path)
            teams = strength["team"].map(normalize_team_name)
            self.fifa_points.update(dict(zip(teams, strength["fifa_points"].fillna(1500))))
            self.elo.update(dict(zip(teams, strength["elo"].fillna(1500))))
            self.last5_form.update(dict(zip(teams, strength["last5_win_rate"].fillna(0.5))))

        ach_path = RAW_DIR / "team_achievements.csv"
        if ach_path.exists():
            ach = pd.read_csv(ach_path)
            ach["team"] = ach["team"].apply(to_wc_team)
            self.achievements = ach.set_index("team")

        pen_path = RAW_DIR / "shootouts.csv"
        if pen_path.exists():
            so = pd.read_csv(pen_path)
            so["winner"] = so["winner"].apply(normalize_team_name)
            so["home_team"] = so["home_team"].apply(normalize_team_name)
            so["away_team"] = so["away_team"].apply(normalize_team_name)
            wins = so["winner"].value_counts()
            games = pd.concat([so["home_team"], so["away_team"]]).value_counts()
            self.penalty_win_rate = (wins / games).fillna(0.5).to_dict()

        # real head-to-head history (NEW: the base simulator left this empty)
        results = load_and_normalize_results()
        self.results = results  # kept for team profiles / h2h detail in the UI
        for home, away, hs, aws in zip(
            results["home_team"], results["away_team"],
            results["home_score"], results["away_score"],
        ):
            key = tuple(sorted([home, away]))
            rec = self.h2h.setdefault(key, {"played": 0, "wins": {}, "draws": 0})
            rec["played"] += 1
            if hs == aws:
                rec["draws"] += 1
            else:
                winner = home if hs > aws else away
                rec["wins"][winner] = rec["wins"].get(winner, 0) + 1

        for team in WC2026_ALL_TEAMS:
            t = normalize_team_name(team)
            self.fifa_points.setdefault(t, 1350)
            self.elo.setdefault(t, 1500)
            self.last5_form.setdefault(t, 0.35)

    def smooth_form(self, team: str, alpha: float = 0.65) -> float:
        # shrink last-5 form toward 0.5
        wr = self.last5_form.get(team, 0.5)
        return alpha * wr + (1 - alpha) * 0.5

    def modern_strength(self, team: str) -> float:
        """Form + tactical bonus + achievement recency decay.
        Used only as a tiebreak/seeding heuristic, never as a model feature."""
        score = (self.smooth_form(team) - 0.5) * 110
        score += MODERN_TEAMS.get(team, 0)
        if self.achievements is not None and team in self.achievements.index:
            row = self.achievements.loc[team]
            for col, weight, decay in [
                ("wc_last_semi_year", 22, 6), ("wc_last_final_year", 28, 6),
                ("wc_last_win_year", 32, 8), ("cont_last_semi_year", 14, 5),
                ("cont_last_final_year", 18, 5), ("cont_last_win_year", 22, 6),
            ]:
                val = row.get(col)
                if pd.notna(val):
                    score += weight * np.exp(-(CURRENT_YEAR - val) / decay)
        return score

    def recent_matches(self, team: str, n: int = 5) -> pd.DataFrame:
        """Last n matches of a team: date, opponent, score, W/D/L result."""
        team = to_wc_team(team)
        df = self.results
        mask = (df["home_team"] == team) | (df["away_team"] == team)
        sub = df[mask].sort_values("date").tail(n).copy()
        rows = []
        for r in sub.itertuples():
            is_home = r.home_team == team
            gf, ga = (r.home_score, r.away_score) if is_home else (r.away_score, r.home_score)
            rows.append({
                "date": r.date.date().isoformat(),
                "opponent": r.away_team if is_home else r.home_team,
                "tournament": r.tournament,
                "score": f"{int(gf)}-{int(ga)}",
                "result": "G" if gf > ga else ("E" if gf == ga else "P"),
            })
        return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)

    def h2h_matches(self, team_a: str, team_b: str, n: int = 8) -> pd.DataFrame:
        """Last n direct meetings between two teams (most recent first)."""
        team_a, team_b = to_wc_team(team_a), to_wc_team(team_b)
        df = self.results
        mask = (
            ((df["home_team"] == team_a) & (df["away_team"] == team_b))
            | ((df["home_team"] == team_b) & (df["away_team"] == team_a))
        )
        sub = df[mask].sort_values("date").tail(n)
        return pd.DataFrame([{
            "date": r.date.date().isoformat(),
            "match": f"{r.home_team} {int(r.home_score)}-{int(r.away_score)} {r.away_team}",
            "tournament": r.tournament,
        } for r in sub.itertuples()]).iloc[::-1].reset_index(drop=True)

    def h2h_stats(self, team_a: str, team_b: str) -> dict:
        # training-consistent defaults: zeros for pairs never seen
        key = tuple(sorted([team_a, team_b]))
        rec = self.h2h.get(key, {"played": 0, "wins": {}, "draws": 0})
        played = rec["played"]
        return {
            "h2h_matches_played": played,
            "h2h_home_win_rate": rec["wins"].get(team_a, 0) / max(played, 1),
            "h2h_draw_rate": rec["draws"] / max(played, 1),
        }

    def base_fifa_diff(self, team_a: str, team_b: str) -> float:
        # FIFA points diff + tactical/squad adjustment
        fifa_diff = self.fifa_points.get(team_a, 1500) - self.fifa_points.get(team_b, 1500)
        tactical = MODERN_TEAMS.get(team_a, 0) - MODERN_TEAMS.get(team_b, 0)
        return fifa_diff + tactical * 12

    def build_features(
        self,
        team_a: str,
        team_b: str,
        *,
        fifa_diff: float | None = None,
        is_friendly: bool = False,
    ) -> dict[str, float]:
        """The 14 model features for an upcoming team_a-vs-team_b match
        (team_a plays the 'home' side; WC matches are neutral anyway)."""
        team_a, team_b = to_wc_team(team_a), to_wc_team(team_b)
        if fifa_diff is None:
            fifa_diff = self.base_fifa_diff(team_a, team_b)
        form_a, form_b = self.smooth_form(team_a), self.smooth_form(team_b)
        h2h = self.h2h_stats(team_a, team_b)
        h2h_weight = min(1.0, h2h["h2h_matches_played"] / 10)
        return {
            "fifa_diff": fifa_diff,
            "elo_diff": self.elo.get(team_a, 1500) - self.elo.get(team_b, 1500),
            "home_last5_win_rate": form_a,
            "away_last5_win_rate": form_b,
            "h2h_home_win_rate": h2h["h2h_home_win_rate"],
            "h2h_draw_rate": h2h["h2h_draw_rate"],
            "h2h_matches_played": h2h["h2h_matches_played"],
            "home_penalty_win_rate": self.penalty_win_rate.get(team_a, 0.5),
            "away_penalty_win_rate": self.penalty_win_rate.get(team_b, 0.5),
            "fifa_diff_x_home_form": fifa_diff * form_a,
            "fifa_diff_x_away_form": fifa_diff * form_b,
            "h2h_effective": h2h["h2h_home_win_rate"] * h2h_weight,
            "is_friendly": int(is_friendly),
            "is_tournament": int(not is_friendly),
        }


@lru_cache(maxsize=1)
def get_live_state() -> LiveState:
    return LiveState()
