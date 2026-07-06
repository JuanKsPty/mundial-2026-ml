"""Walk-forward (expanding window) backtest of classifier + Poisson models.

For each year Y:  train < Y-2  |  calibrate [Y-2, Y)  |  test == Y
Run: python -m src.evaluation.backtest
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

from src.config import FIGURES_DIR, OUTPUTS_DIR
from src.models.match_model import (
    FEATURE_COLS,
    _load_features,
    _sample_weights,
    multiclass_brier,
)
from src.models.poisson_model import (
    POISSON_FEATURE_COLS,
    LAMBDA_MIN,
    LAMBDA_MAX,
    estimate_rho,
    matrix_outcome_probs,
    most_likely_score,
    score_matrix,
)
from src.utils import viz
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def rps(probs_hda: np.ndarray, outcome_hda: int) -> float:
    """Ranked probability score for one match. probs ordered [home, draw, away]
    (an ordinal scale); outcome_hda: 0=home, 1=draw, 2=away. Lower is better."""
    onehot = np.zeros(3)
    onehot[outcome_hda] = 1.0
    cum_diff = np.cumsum(probs_hda) - np.cumsum(onehot)
    return float(np.sum(cum_diff[:-1] ** 2) / 2)


def _fit_window(train: pd.DataFrame, calib: pd.DataFrame, feature_cols, poisson_cols):
    """Fit GBM + sigmoid calibration + Poisson models on one window."""
    X_train = train[feature_cols].fillna(0)
    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gbm.fit(X_train, train["outcome"], sample_weight=_sample_weights(train))

    # sigmoid: the method selected on the main model; isotonic overfits the
    # smaller per-window calibration sets
    cal = CalibratedClassifierCV(gbm, method="sigmoid", cv="prefit")
    cal.fit(calib[feature_cols].fillna(0), calib["outcome"],
            sample_weight=_sample_weights(calib))

    fit_df = pd.concat([train, calib])
    fit_df = fit_df[(fit_df["home_score"] <= 10) & (fit_df["away_score"] <= 10)]
    Xp = fit_df[poisson_cols].fillna(0)
    p_home = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=300))
    p_away = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=300))
    p_home.fit(Xp, fit_df["home_score"])
    p_away.fit(Xp, fit_df["away_score"])
    lam_h = np.clip(p_home.predict(Xp), LAMBDA_MIN, LAMBDA_MAX)
    lam_a = np.clip(p_away.predict(Xp), LAMBDA_MIN, LAMBDA_MAX)
    rho = estimate_rho(fit_df, lam_h, lam_a)
    return cal, p_home, p_away, rho


def _eval_year(test, cal, p_home, p_away, rho, feature_cols, poisson_cols) -> dict:
    X_test = test[feature_cols].fillna(0)
    y = test["outcome"].to_numpy()
    proba = cal.predict_proba(X_test)  # columns: [away(0), draw(1), home(2)]

    # ordinal order [home, draw, away] for RPS; outcome 2->0, 1->1, 0->2
    probs_hda = proba[:, [2, 1, 0]]
    out_hda = 2 - y
    rps_mean = float(np.mean([rps(p, o) for p, o in zip(probs_hda, out_hda)]))

    lam_h = np.clip(p_home.predict(test[poisson_cols].fillna(0)), LAMBDA_MIN, LAMBDA_MAX)
    lam_a = np.clip(p_away.predict(test[poisson_cols].fillna(0)), LAMBDA_MIN, LAMBDA_MAX)
    hs = test["home_score"].to_numpy(dtype=int)
    aws = test["away_score"].to_numpy(dtype=int)

    exact_hits = 0
    outcome_hits_poisson = 0
    for lh, la, h, a in zip(lam_h, lam_a, hs, aws):
        m = score_matrix(lh, la, rho)
        (ph, pa), _ = most_likely_score(m)
        exact_hits += int((ph, pa) == (h, a))
        p_h, p_d, p_a = matrix_outcome_probs(m)
        pred = int(np.argmax([p_a, p_d, p_h]))  # same 0/1/2 encoding as outcome
        actual = 2 if h > a else (0 if h < a else 1)
        outcome_hits_poisson += int(pred == actual)

    return {
        "n_test": len(test),
        "clf_accuracy": float(accuracy_score(y, np.argmax(proba, axis=1))),
        "clf_log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
        "clf_brier": multiclass_brier(y, proba),
        "clf_rps": rps_mean,
        "poisson_exact_score_acc": exact_hits / len(test),
        "poisson_outcome_acc": outcome_hits_poisson / len(test),
        "poisson_goals_mae": float(np.mean(np.abs(lam_h - hs)) + np.mean(np.abs(lam_a - aws))) / 2,
    }


def run_backtest(start_year: int = 2018, end_year: int = 2025,
                 calib_window: int = 2) -> pd.DataFrame:
    """Expanding-window backtest. end_year=2025: WC2026 stays out of metrics."""
    df = _load_features()
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    poisson_cols = [c for c in POISSON_FEATURE_COLS if c in df.columns]

    rows = []
    for year in range(start_year, end_year + 1):
        train = df[df["year"] < year - calib_window]
        calib = df[(df["year"] >= year - calib_window) & (df["year"] < year)]
        test = df[df["year"] == year]
        if test.empty or len(train) < 1000:
            continue
        print(f"  {year}: train={len(train)}  calib={len(calib)}  test={len(test)} ...", flush=True)
        cal, p_home, p_away, rho = _fit_window(train, calib, feature_cols, poisson_cols)
        metrics = _eval_year(test, cal, p_home, p_away, rho, feature_cols, poisson_cols)
        rows.append({"year": year, "rho": rho, **metrics})

    result = pd.DataFrame(rows)
    weights = result["n_test"] / result["n_test"].sum()
    mean_row = {"year": "weighted_mean", "n_test": int(result["n_test"].sum())}
    for col in result.columns:
        if col not in ("year", "n_test"):
            mean_row[col] = float((result[col] * weights).sum())
    result = pd.concat([result, pd.DataFrame([mean_row])], ignore_index=True)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "backtest_results.csv"
    result.to_csv(out_path, index=False)
    print(f"  [OK] Backtest results -> {out_path}")
    _plot_backtest(result[result["year"] != "weighted_mean"])
    return result


def _plot_backtest(df: pd.DataFrame) -> None:
    viz.apply_style()
    panels = [
        ("clf_accuracy", "Accuracy W/D/L (calibrated)", viz.BLUE),
        ("clf_log_loss", "Log-loss (lower = better)", viz.BLUE),
        ("clf_rps", "RPS (lower = better)", viz.BLUE),
        ("poisson_exact_score_acc", "Exact score accuracy (Poisson)", viz.AQUA),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    years = df["year"].astype(int)
    for ax, (col, title, color) in zip(axes.ravel(), panels):
        ax.plot(years, df[col], "o-", color=color, markersize=5)
        ax.set_title(title, fontsize=10, color=viz.INK_SECONDARY, loc="left")
        ax.set_xticks(years)
        # direct label on the last point only
        ax.annotate(f"{df[col].iloc[-1]:.3f}", (years.iloc[-1], df[col].iloc[-1]),
                    textcoords="offset points", xytext=(6, -3),
                    fontsize=8, color=viz.INK_SECONDARY)
    fig.suptitle("Walk-forward backtest (expanding window, test = each year)",
                 fontsize=12)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "backtest_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [OK] Backtest figure -> {out}")


if __name__ == "__main__":
    print("Running walk-forward backtest...")
    res = run_backtest()
    cols = ["year", "n_test", "clf_accuracy", "clf_log_loss", "clf_brier", "clf_rps",
            "poisson_exact_score_acc", "poisson_outcome_acc", "poisson_goals_mae"]
    print(res[cols].to_string(index=False))
