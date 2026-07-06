# Design notes:
#   * chronological train/calibration/test split (a random split would be
#     optimistic for time-series data).
#   * probability calibration with CalibratedClassifierCV (isotonic vs
#     sigmoid chosen by Brier score on the held-out test years).
#   * calibration evidence: reliability curves + Brier/log-loss report.

"""W/D/L outcome model: sklearn GradientBoosting + calibrated probabilities."""

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss

from src.config import (
    CALIB_START_YEAR,
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    TEST_START_YEAR,
    TRAIN_END_YEAR,
    TRAIN_MAX_DATE,
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["away", "draw", "home"]  # outcome: 0=away win, 1=draw, 2=home win

FEATURE_COLS = [
    "fifa_diff", "elo_diff",
    "home_last5_win_rate", "away_last5_win_rate",
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_matches_played",
    "home_penalty_win_rate", "away_penalty_win_rate",
    "fifa_diff_x_home_form", "fifa_diff_x_away_form",
    "h2h_effective", "is_friendly", "is_tournament",
]


def _load_features(features_path=None) -> pd.DataFrame:
    path = features_path or PROCESSED_DIR / "match_features.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    # hard cutoff: the ongoing WC2026 must never leak into training/eval
    df = df[df["date"] <= pd.Timestamp(TRAIN_MAX_DATE)].copy()
    df["year"] = df["date"].dt.year
    return df


def _temporal_split(df: pd.DataFrame):
    train = df[df["year"] <= TRAIN_END_YEAR]
    calib = df[(df["year"] >= CALIB_START_YEAR) & (df["year"] < TEST_START_YEAR)]
    test = df[df["year"] >= TEST_START_YEAR]
    return train, calib, test


def _sample_weights(df: pd.DataFrame) -> np.ndarray:
    # weight competitive matches higher than friendlies
    weights = np.ones(len(df))
    weights += df["is_tournament"].to_numpy() * 2
    weights += (1 - df["is_friendly"].to_numpy()) * 0.5
    return weights


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> float:
    """Mean squared error between one-hot outcomes and predicted probabilities."""
    onehot = np.eye(n_classes)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _eval_probs(y_true, proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, np.argmax(proba, axis=1))),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
        "brier": multiclass_brier(y_true, proba),
    }


def train_match_model(
    features_path: str = None,
    sample_weight_tournaments: bool = True,
) -> dict:
    """Train GBM on the chronological train years, calibrate on the calibration
    years, evaluate raw-vs-calibrated on the test years. Returns the bundle."""
    print("Training match outcome model (Gradient Boosting + calibration)...")

    df = _load_features(features_path)
    available = [c for c in FEATURE_COLS if c in df.columns]
    train, calib, test = _temporal_split(df)
    print(
        f"  Temporal split: train<= {TRAIN_END_YEAR} ({len(train)}), "
        f"calib {CALIB_START_YEAR}-{TEST_START_YEAR - 1} ({len(calib)}), "
        f"test >= {TEST_START_YEAR} ({len(test)}, cutoff {TRAIN_MAX_DATE})"
    )

    X_train = train[available].fillna(0)
    y_train = train["outcome"]
    X_calib = calib[available].fillna(0)
    y_calib = calib["outcome"]
    X_test = test[available].fillna(0)
    y_test = test["outcome"]

    w_train = _sample_weights(train) if sample_weight_tournaments else None
    w_calib = _sample_weights(calib) if sample_weight_tournaments else None

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    # --- calibration: fit isotonic and sigmoid on the calibration years,
    #     keep whichever has the lower Brier score on the test years ---
    metrics = {"raw": _eval_probs(y_test, model.predict_proba(X_test))}
    calibrators = {}
    for method in ("isotonic", "sigmoid"):
        cal = CalibratedClassifierCV(model, method=method, cv="prefit")
        cal.fit(X_calib, y_calib, sample_weight=w_calib)
        calibrators[method] = cal
        metrics[method] = _eval_probs(y_test, cal.predict_proba(X_test))

    best_method = min(("isotonic", "sigmoid"), key=lambda m: metrics[m]["brier"])
    calibrator = calibrators[best_method]

    for name in ("raw", "isotonic", "sigmoid"):
        m = metrics[name]
        marker = " <- selected" if name == best_method else ""
        print(
            f"  [{name:8s}] acc={m['accuracy']:.3f}  log_loss={m['log_loss']:.4f}  "
            f"brier={m['brier']:.4f}{marker}"
        )
    print(classification_report(
        y_test, calibrator.predict(X_test), target_names=CLASS_NAMES,
    ))

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "calibration_method": best_method,
        "feature_cols": available,
        "metrics": metrics,
        "split": {
            "train_end_year": TRAIN_END_YEAR,
            "calib_start_year": CALIB_START_YEAR,
            "test_start_year": TEST_START_YEAR,
            "train_max_date": TRAIN_MAX_DATE,
            "n_train": len(train), "n_calib": len(calib), "n_test": len(test),
        },
    }

    save_path = MODELS_DIR / "match_outcome.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  [OK] Model bundle saved -> {save_path}")

    report = pd.DataFrame(metrics).T.reset_index(names="probabilities")
    report_path = OUTPUTS_DIR / "calibration_report.csv"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    print(f"  [OK] Calibration report -> {report_path}")
    return bundle


def load_match_model() -> dict | None:
    path = MODELS_DIR / "match_outcome.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_proba_calibrated(bundle: dict, X) -> np.ndarray:
    """Calibrated [p_away, p_draw, p_home] rows for a feature matrix."""
    return bundle["calibrator"].predict_proba(X)


def plot_calibration(bundle: dict = None, features_path: str = None) -> str:
    """Reliability curves per class, raw GBM vs calibrated, on the test years."""
    bundle = bundle or load_match_model()
    if bundle is None:
        raise FileNotFoundError("Train the model first (python -m src.models.match_model)")

    df = _load_features(features_path)
    _, _, test = _temporal_split(df)
    X_test = test[bundle["feature_cols"]].fillna(0)
    y_test = test["outcome"].to_numpy()

    proba_raw = bundle["model"].predict_proba(X_test)
    proba_cal = bundle["calibrator"].predict_proba(X_test)

    from src.utils import viz
    viz.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for k, (ax, cls) in enumerate(zip(axes, CLASS_NAMES)):
        y_bin = (y_test == k).astype(int)
        for proba, label, style, color in (
            (proba_raw, "GBM raw", "o--", viz.RED),
            (proba_cal, f"calibrated ({bundle['calibration_method']})", "s-", viz.BLUE),
        ):
            frac_pos, mean_pred = calibration_curve(y_bin, proba[:, k], n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, style, label=label, alpha=0.9, color=color)
        ax.plot([0, 1], [0, 1], ":", color=viz.INK_MUTED, alpha=0.7, label="perfect")
        ax.set_title(f"class: {cls}")
        ax.set_xlabel("mean predicted probability")
        if k == 0:
            ax.set_ylabel("observed frequency")
        ax.legend(fontsize=8)
    fig.suptitle("Calibration curves on temporal test set "
                 f"({TEST_START_YEAR}+, n={len(y_test)})")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "calibration_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [OK] Calibration figure -> {out}")
    return str(out)


if __name__ == "__main__":
    train_match_model()
    plot_calibration()
