"""Central prediction: team1, team2 -> result + exact score + probabilities.

CLI:  python -m src.predict "Argentina" "France"
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import WC2026_ALL_TEAMS
from src.features.live_features import get_live_state, to_wc_team
from src.models.match_model import load_match_model, predict_proba_calibrated
from src.models.poisson_model import (
    load_poisson_models,
    most_likely_score,
    predict_lambdas,
    score_matrix,
    top_scores,
)

RESULT_LABELS = {"H": "home win", "D": "draw", "A": "away win"}


@dataclass
class MatchPrediction:
    home_team: str
    away_team: str
    probabilities: dict            # {"home": p, "draw": p, "away": p} calibrated
    predicted_result: str          # "H" | "D" | "A" (argmax of calibrated probs)
    expected_goals: tuple          # (lambda_home, lambda_away)
    predicted_score: tuple         # most likely score consistent with the result
    predicted_score_prob: float    # prob of that score within the result region
    top_scores: list = field(default_factory=list)  # [( (h,a), p ), ...] unconditioned
    score_matrix: np.ndarray | None = None          # (MAX_GOALS+1)^2, sums to 1

    def summary(self) -> str:
        p = self.probabilities
        lines = [
            f"{self.home_team} vs {self.away_team}",
            f"  P(win {self.home_team}): {p['home']:.1%}   P(draw): {p['draw']:.1%}   "
            f"P(win {self.away_team}): {p['away']:.1%}",
            f"  predicted result : {RESULT_LABELS[self.predicted_result]}",
            f"  expected goals   : {self.expected_goals[0]:.2f} - {self.expected_goals[1]:.2f}",
            f"  predicted score  : {self.predicted_score[0]}-{self.predicted_score[1]} "
            f"({self.predicted_score_prob:.1%} within predicted result)",
            "  top scores       : " + "   ".join(
                f"{h}-{a} {prob:.1%}" for (h, a), prob in self.top_scores
            ),
        ]
        return "\n".join(lines)


def _validate_team(name: str) -> str:
    team = to_wc_team(name)
    if team not in WC2026_ALL_TEAMS:
        import difflib
        close = difflib.get_close_matches(name, WC2026_ALL_TEAMS, n=3, cutoff=0.4)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ValueError(f"'{name}' is not a WC2026 team.{hint}")
    return team


def predict_match(
    home: str, away: str, *, is_friendly: bool = False, neutral: bool = True,
) -> MatchPrediction:
    """Predict an upcoming match between two WC2026 teams.

    With neutral=True (default; WC matches are on neutral ground) the model is
    evaluated in both orientations and averaged, so the training data's
    home-side bias cancels out and predict(A, B) mirrors predict(B, A)."""
    home, away = _validate_team(home), _validate_team(away)

    clf_bundle = load_match_model()
    poisson_bundle = load_poisson_models()
    if clf_bundle is None or poisson_bundle is None:
        raise FileNotFoundError(
            "Models not trained - run: python -m scripts.build_all"
        )

    state = get_live_state()
    feats = state.build_features(home, away, is_friendly=is_friendly)
    frames = [feats]
    if neutral:
        frames.append(state.build_features(away, home, is_friendly=is_friendly))
    X = pd.DataFrame(frames)[clf_bundle["feature_cols"]]

    proba = predict_proba_calibrated(clf_bundle, X)  # rows of [away, draw, home]
    if neutral:
        p_home = (proba[0, 2] + proba[1, 0]) / 2
        p_draw = (proba[0, 1] + proba[1, 1]) / 2
        p_away = (proba[0, 0] + proba[1, 2]) / 2
        total = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total
    else:
        p_away, p_draw, p_home = proba[0]
    predicted_result = {0: "A", 1: "D", 2: "H"}[int(np.argmax([p_away, p_draw, p_home]))]

    lam_home, lam_away = predict_lambdas(poisson_bundle, feats)
    if neutral:
        lam_home_rev, lam_away_rev = predict_lambdas(poisson_bundle, frames[1])
        lam_home = (lam_home + lam_away_rev) / 2
        lam_away = (lam_away + lam_home_rev) / 2
    matrix = score_matrix(lam_home, lam_away, poisson_bundle["rho"])
    # consistency rule: the displayed score always agrees with the predicted
    # result (argmax within that region of the score matrix)
    score, score_prob = most_likely_score(matrix, condition=predicted_result)

    return MatchPrediction(
        home_team=home,
        away_team=away,
        probabilities={"home": float(p_home), "draw": float(p_draw), "away": float(p_away)},
        predicted_result=predicted_result,
        expected_goals=(lam_home, lam_away),
        predicted_score=score,
        predicted_score_prob=score_prob,
        top_scores=top_scores(matrix, 5),
        score_matrix=matrix,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict a WC2026 match")
    parser.add_argument("home", help="first team (e.g. Argentina)")
    parser.add_argument("away", help="second team (e.g. France)")
    args = parser.parse_args()
    print(predict_match(args.home, args.away).summary())
