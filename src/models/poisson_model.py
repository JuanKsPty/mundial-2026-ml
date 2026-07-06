# Expected-goals model for exact scores: two sklearn PoissonRegressor
# (home/away goals) over the same walk-forward features as the classifier,
# plus the Dixon-Coles tau adjustment for the dependence between goals in
# low-scoring games (rho fitted by grid search).
# Conceptual reference: Dixon & Coles (1997).

"""Poisson expected-goals model: score matrix, exact score, score sampling."""

import math
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MAX_GOALS, MODELS_DIR, PROCESSED_DIR, TRAIN_MAX_DATE

MODELS_DIR.mkdir(parents=True, exist_ok=True)

POISSON_FEATURE_COLS = [
    "fifa_diff", "elo_diff",
    "home_last5_win_rate", "away_last5_win_rate",
    "h2h_home_win_rate", "h2h_draw_rate",
    "is_friendly", "is_tournament",
]

LAMBDA_MIN, LAMBDA_MAX = 0.05, 4.5  # clip expected goals to a sane range

# outcome regions of the score matrix (i=home goals, j=away goals)
_CONDITION_MASKS = {}


def _load_features(features_path=None, max_date: str = TRAIN_MAX_DATE) -> pd.DataFrame:
    path = features_path or PROCESSED_DIR / "match_features.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(max_date)].copy()
    # very rare extreme scores (e.g. 31-0 vs micro-states) distort the fit
    df = df[(df["home_score"] <= 10) & (df["away_score"] <= 10)]
    return df


def _poisson_pmf_vector(lam: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    ks = np.arange(max_goals + 1)
    pmf = np.exp(-lam) * lam ** ks / np.array([math.factorial(k) for k in ks])
    return pmf


def dixon_coles_tau(i: int, j: int, lam: float, mu: float, rho: float) -> float:
    """Low-score dependence adjustment (Dixon & Coles 1997)."""
    if i == 0 and j == 0:
        return 1.0 - lam * mu * rho
    if i == 1 and j == 0:
        return 1.0 + mu * rho
    if i == 0 and j == 1:
        return 1.0 + lam * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lam_home: float, lam_away: float, rho: float = 0.0,
                 max_goals: int = MAX_GOALS) -> np.ndarray:
    """P(home=i, away=j) for i,j in 0..max_goals, Dixon-Coles adjusted,
    renormalized to sum 1. Row index = home goals, column = away goals."""
    lam_home = float(np.clip(lam_home, LAMBDA_MIN, LAMBDA_MAX))
    lam_away = float(np.clip(lam_away, LAMBDA_MIN, LAMBDA_MAX))
    m = np.outer(_poisson_pmf_vector(lam_home, max_goals),
                 _poisson_pmf_vector(lam_away, max_goals))
    for i, j in ((0, 0), (1, 0), (0, 1), (1, 1)):
        m[i, j] *= max(dixon_coles_tau(i, j, lam_home, lam_away, rho), 1e-10)
    return m / m.sum()


def _condition_mask(condition: str | None, n: int) -> np.ndarray:
    """Boolean mask of the matrix region for an outcome: H (home win),
    D (draw), A (away win) or None (whole matrix)."""
    key = (condition, n)
    if key not in _CONDITION_MASKS:
        i, j = np.indices((n, n))
        if condition == "H":
            mask = i > j
        elif condition == "A":
            mask = i < j
        elif condition == "D":
            mask = i == j
        elif condition is None:
            mask = np.ones((n, n), dtype=bool)
        else:
            raise ValueError(f"unknown condition: {condition}")
        _CONDITION_MASKS[key] = mask
    return _CONDITION_MASKS[key]


def matrix_outcome_probs(m: np.ndarray) -> tuple[float, float, float]:
    """(p_home, p_draw, p_away) implied by the score matrix."""
    n = m.shape[0]
    return (
        float(m[_condition_mask("H", n)].sum()),
        float(m[_condition_mask("D", n)].sum()),
        float(m[_condition_mask("A", n)].sum()),
    )


def most_likely_score(m: np.ndarray, condition: str | None = None) -> tuple[tuple[int, int], float]:
    """((home_goals, away_goals), prob). With a condition, the argmax is taken
    inside that outcome region and the prob is renormalized within it."""
    mask = _condition_mask(condition, m.shape[0])
    region = np.where(mask, m, -1.0)
    i, j = np.unravel_index(int(np.argmax(region)), m.shape)
    prob = float(m[i, j] / m[mask].sum()) if condition else float(m[i, j])
    return (int(i), int(j)), prob


def top_scores(m: np.ndarray, k: int = 5) -> list[tuple[tuple[int, int], float]]:
    flat = np.argsort(m, axis=None)[::-1][:k]
    return [(tuple(int(x) for x in np.unravel_index(f, m.shape)), float(m.flat[f]))
            for f in flat]


def sample_score(m: np.ndarray, rng: np.random.Generator,
                 condition: str | None = None) -> tuple[int, int]:
    """Draw one (home_goals, away_goals) from the matrix, optionally
    conditioned on an outcome region (used by the Monte Carlo simulator)."""
    mask = _condition_mask(condition, m.shape[0])
    probs = np.where(mask, m, 0.0).ravel()
    probs = probs / probs.sum()
    flat = rng.choice(probs.size, p=probs)
    i, j = np.unravel_index(int(flat), m.shape)
    return int(i), int(j)


def estimate_rho(df: pd.DataFrame, lam_home: np.ndarray, lam_away: np.ndarray,
                 grid: np.ndarray | None = None) -> float:
    """Grid-search rho maximizing the Dixon-Coles log-likelihood of the
    observed scores given the fitted lambdas. Only the four low-score cells
    depend on rho, so the sweep is cheap."""
    if grid is None:
        grid = np.round(np.arange(-0.2, 0.201, 0.02), 3)
    hs = df["home_score"].to_numpy(dtype=int)
    aws = df["away_score"].to_numpy(dtype=int)

    base_ll = (
        -lam_home + hs * np.log(lam_home) - np.array([math.lgamma(h + 1) for h in hs])
        - lam_away + aws * np.log(lam_away) - np.array([math.lgamma(a + 1) for a in aws])
    )

    best_rho, best_ll = 0.0, -np.inf
    low = (hs <= 1) & (aws <= 1)
    for rho in grid:
        tau = np.ones(len(df))
        tau[low] = [
            dixon_coles_tau(h, a, lh, la, rho)
            for h, a, lh, la in zip(hs[low], aws[low], lam_home[low], lam_away[low])
        ]
        if (tau <= 0).any():
            continue
        ll = float(np.sum(base_ll + np.log(tau)))
        if ll > best_ll:
            best_ll, best_rho = ll, float(rho)
    return best_rho


def train_poisson_models(features_path: str = None, max_date: str = TRAIN_MAX_DATE,
                         save: bool = True) -> dict:
    """Fit home/away PoissonRegressor + Dixon-Coles rho. Returns the bundle."""
    print("Training Poisson expected-goals models...")
    df = _load_features(features_path, max_date)
    available = [c for c in POISSON_FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)

    # StandardScaler keeps the solver stable (fifa_diff is on a scale of
    # hundreds of ranking points, the rest are rates in [0, 1])
    home_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=300))
    away_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=300))
    home_model.fit(X, df["home_score"])
    away_model.fit(X, df["away_score"])

    lam_home = np.clip(home_model.predict(X), LAMBDA_MIN, LAMBDA_MAX)
    lam_away = np.clip(away_model.predict(X), LAMBDA_MIN, LAMBDA_MAX)
    rho = estimate_rho(df, lam_home, lam_away)

    mae_home = float(np.mean(np.abs(lam_home - df["home_score"])))
    mae_away = float(np.mean(np.abs(lam_away - df["away_score"])))
    print(f"  n={len(df)}  mean lambda home={lam_home.mean():.2f} away={lam_away.mean():.2f}")
    print(f"  goals MAE: home={mae_home:.3f} away={mae_away:.3f}  |  Dixon-Coles rho={rho:+.3f}")

    bundle = {
        "home_model": home_model,
        "away_model": away_model,
        "rho": rho,
        "feature_cols": available,
        "metrics": {"mae_home": mae_home, "mae_away": mae_away, "n_train": len(df)},
        "max_date": max_date,
    }
    if save:
        path = MODELS_DIR / "poisson_goals.pkl"
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"  [OK] Poisson bundle saved -> {path}")
    return bundle


def load_poisson_models() -> dict | None:
    path = MODELS_DIR / "poisson_goals.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_lambdas(bundle: dict, features: dict | pd.DataFrame) -> tuple[float, float]:
    """Expected goals (lam_home, lam_away) for one feature dict/row."""
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    X = features[bundle["feature_cols"]].fillna(0)
    lam_h = float(np.clip(bundle["home_model"].predict(X)[0], LAMBDA_MIN, LAMBDA_MAX))
    lam_a = float(np.clip(bundle["away_model"].predict(X)[0], LAMBDA_MIN, LAMBDA_MAX))
    return lam_h, lam_a


if __name__ == "__main__":
    b = train_poisson_models()
    m = score_matrix(1.8, 0.9, b["rho"])
    print("\nSanity check lam=(1.8, 0.9):")
    print("  outcome probs (H,D,A):", tuple(round(p, 3) for p in matrix_outcome_probs(m)))
    print("  most likely score:", most_likely_score(m))
    print("  top-5:", [(s, round(p, 3)) for s, p in top_scores(m)])
