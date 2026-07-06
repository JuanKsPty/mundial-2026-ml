"""End-to-end pipeline: data -> features -> models -> figures -> simulations.

Run: python -m scripts.build_all [--simulations 5000] [--use-cache] [--skip-backtest]
"""

import argparse
import json
from datetime import datetime, timezone

from src.config import N_SIMULATIONS_DEFAULT, OUTPUTS_DIR
from src.data.historical_data import (
    build_team_achievements,
    download_martj42_dataset,
    fetch_fifa_rankings,
)
from src.features.match_features import build_match_features, build_team_strength_table
from src.models.match_model import plot_calibration, train_match_model
from src.models.poisson_model import train_poisson_models


def run_pipeline(
    n_simulations: int = N_SIMULATIONS_DEFAULT,
    *,
    refresh_data: bool = True,
    n_workers: int | None = None,
    run_backtest: bool = True,
) -> None:
    print("\n[1/7] Historical data (martj42 + FIFA rankings)...")
    download_martj42_dataset(force=refresh_data)
    fetch_fifa_rankings(force=refresh_data)
    build_team_achievements()

    print("\n[2/7] Match features + team strength...")
    build_match_features()
    build_team_strength_table()

    print("\n[3/7] W/D/L model (GBM + temporal calibration)...")
    train_match_model()
    plot_calibration()

    print("\n[4/7] Poisson expected-goals model...")
    train_poisson_models()

    print("\n[5/7] Syncing WC2026 matches -> SQLite...")
    from scripts.sync_wc2026 import main as sync_main
    sync_main(source="auto")

    if run_backtest:
        print("\n[6/7] Walk-forward backtest...")
        from src.evaluation.backtest import run_backtest as run_bt
        run_bt()
    else:
        print("\n[6/7] Backtest skipped (--skip-backtest)")

    print("\n[7/7] Monte Carlo simulations...")
    from src.simulation.montecarlo import run_full_simulation
    run_full_simulation(n_simulations, n_workers=n_workers)
    try:
        from src.simulation.tournament_state import simulate_from_current_state
        simulate_from_current_state(n_simulations)
    except FileNotFoundError as e:
        print(f"  [WARN] current-state simulation skipped: {e}")

    meta = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_simulations": n_simulations,
        "refresh_data": refresh_data,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "build_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\n[OK] Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WC2026 predictor full pipeline")
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS_DEFAULT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--use-cache", action="store_true",
        help="refresh martj42/FIFA only when older than the TTL",
    )
    parser.add_argument("--skip-backtest", action="store_true",
                        help="skip the walk-forward backtest (slowest step)")
    args = parser.parse_args()
    run_pipeline(
        args.simulations,
        refresh_data=not args.use_cache,
        n_workers=args.workers,
        run_backtest=not args.skip_backtest,
    )
