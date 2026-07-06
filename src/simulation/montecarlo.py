# Design notes:
#   * outcomes are SAMPLED from the calibrated multinomial probabilities.
#   * scores are sampled from the Dixon-Coles Poisson matrix conditioned on
#     the sampled outcome -> group tables use REAL simulated goals.
#   * neutral-venue symmetrization: each pairing is predicted in both
#     orientations and averaged.
#   * per-pair predictions are precomputed/cached (features are static within
#     a simulation run; only sampling varies).
#   * per-round score noise: lambdas are jittered lognormally (ROUND_GOAL_NOISE)
#     instead of noising the rating diff.
#   * groups: top-2 + 8 best thirds; knockout draws go to penalty shootouts
#     via historical shootout win rates; ProcessPoolExecutor batching.
#   * the full-tournament mode uses a random knockout draw (documented
#     limitation); the current-state mode (tournament_state.py) uses the
#     real bracket instead.

"""Monte Carlo simulation of the WC2026 tournament (full mode)."""

from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    N_SIMULATIONS_DEFAULT,
    OUTPUTS_DIR,
    ROUND_GOAL_NOISE,
    WC2026_ALL_TEAMS,
    WC2026_GROUPS,
    WC2026_TEAM_TO_GROUP,
)
from src.features.live_features import LiveState, get_live_state, to_wc_team
from src.models.match_model import load_match_model
from src.models.poisson_model import (
    LAMBDA_MAX,
    LAMBDA_MIN,
    load_poisson_models,
    sample_score,
    score_matrix,
)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn.base",
)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

KO_ROUNDS = ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"]
# progression counters tracked per team (reaching each stage + winning it all)
STAGES = ["r32", "r16", "qf", "sf", "final", "champion"]
_ROUND_TO_STAGE = {
    "round_of_32": "r32", "round_of_16": "r16", "quarterfinal": "qf",
    "semifinal": "sf", "final": "final",
}


@dataclass
class MatchSampler:
    """Cached, symmetrized (neutral-venue) predictions for team pairings,
    plus outcome/score sampling."""

    clf_bundle: dict
    poisson_bundle: dict
    state: LiveState
    _cache: dict = field(default_factory=dict)

    def prediction(self, team_a: str, team_b: str) -> tuple[np.ndarray, float, float]:
        """(probs [p_a_win, p_draw, p_b_win], lam_a, lam_b) - order-invariant."""
        key = (team_a, team_b)
        if key in self._cache:
            return self._cache[key]
        rev = self._cache.get((team_b, team_a))
        if rev is not None:
            probs, lam_b, lam_a = rev
            out = (probs[::-1].copy(), lam_a, lam_b)
            self._cache[key] = out
            return out

        feats_ab = self.state.build_features(team_a, team_b)
        feats_ba = self.state.build_features(team_b, team_a)
        cols = self.clf_bundle["feature_cols"]
        X = pd.DataFrame([feats_ab, feats_ba])[cols].fillna(0)
        p = self.clf_bundle["calibrator"].predict_proba(X)  # [away, draw, home]

        # average both orientations: p(A wins) = mean(home(A|AB), away(A|BA))
        p_a = (p[0, 2] + p[1, 0]) / 2
        p_d = (p[0, 1] + p[1, 1]) / 2
        p_b = (p[0, 0] + p[1, 2]) / 2
        probs = np.array([p_a, p_d, p_b]) / (p_a + p_d + p_b)

        pcols = self.poisson_bundle["feature_cols"]
        Xp = pd.DataFrame([feats_ab, feats_ba])[pcols].fillna(0)
        lh = np.clip(self.poisson_bundle["home_model"].predict(Xp), LAMBDA_MIN, LAMBDA_MAX)
        la = np.clip(self.poisson_bundle["away_model"].predict(Xp), LAMBDA_MIN, LAMBDA_MAX)
        lam_a = (lh[0] + la[1]) / 2
        lam_b = (la[0] + lh[1]) / 2

        out = (probs, float(lam_a), float(lam_b))
        self._cache[key] = out
        return out

    def sample_match(
        self, team_a: str, team_b: str, rng: np.random.Generator, round_name: str,
    ) -> tuple[int, int]:
        """Sample (goals_a, goals_b): outcome ~ calibrated multinomial, then
        score ~ Poisson matrix conditioned on that outcome."""
        probs, lam_a, lam_b = self.prediction(team_a, team_b)
        outcome = ("H", "D", "A")[rng.choice(3, p=probs)]

        sigma = ROUND_GOAL_NOISE.get(round_name, 0.1)
        lam_a = lam_a * float(np.exp(rng.normal(0, sigma)))
        lam_b = lam_b * float(np.exp(rng.normal(0, sigma)))
        matrix = score_matrix(lam_a, lam_b, self.poisson_bundle["rho"])
        return sample_score(matrix, rng, condition=outcome)

    def sample_knockout(
        self, team_a: str, team_b: str, rng: np.random.Generator, round_name: str,
    ) -> tuple[str, tuple[int, int], bool]:
        """(winner, (goals_a, goals_b), went_to_penalties)."""
        goals_a, goals_b = self.sample_match(team_a, team_b, rng, round_name)
        if goals_a > goals_b:
            return team_a, (goals_a, goals_b), False
        if goals_b > goals_a:
            return team_b, (goals_a, goals_b), False
        # draw -> penalty shootout, via historical shootout win rates
        pa = self.state.penalty_win_rate.get(team_a, 0.5)
        pb = self.state.penalty_win_rate.get(team_b, 0.5)
        winner = team_a if rng.random() < pa / (pa + pb) else team_b
        return winner, (goals_a, goals_b), True


def make_sampler() -> MatchSampler:
    clf = load_match_model()
    poisson = load_poisson_models()
    if clf is None or poisson is None:
        raise FileNotFoundError("Models not trained - run: python -m scripts.build_all")
    return MatchSampler(clf_bundle=clf, poisson_bundle=poisson, state=get_live_state())


def warm_group_cache(sampler: MatchSampler) -> None:
    """Precompute the 72 fixed group pairings once (before forking workers)."""
    for teams in WC2026_GROUPS.values():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                sampler.prediction(teams[i], teams[j])


def simulate_group(
    teams: list[str], sampler: MatchSampler, rng: np.random.Generator,
) -> list[tuple[str, dict]]:
    """Round-robin with real sampled scores; ranked by points, gd, gf."""
    table = {t: {"points": 0, "gd": 0, "gf": 0} for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            ga, gb = sampler.sample_match(a, b, rng, "group")
            table[a]["gf"] += ga
            table[a]["gd"] += ga - gb
            table[b]["gf"] += gb
            table[b]["gd"] += gb - ga
            if ga > gb:
                table[a]["points"] += 3
            elif gb > ga:
                table[b]["points"] += 3
            else:
                table[a]["points"] += 1
                table[b]["points"] += 1
    return sorted(
        table.items(),
        key=lambda kv: (kv[1]["points"], kv[1]["gd"], kv[1]["gf"], rng.random()),
        reverse=True,
    )


def simulate_one_tournament(
    sampler: MatchSampler, rng: np.random.Generator,
) -> tuple[dict[str, str], dict[str, dict]]:
    """One full tournament. Returns (stage_reached per team, group stats)."""
    stage_reached: dict[str, str] = {}
    group_stats = {
        team: {"points": 0.0, "top_group": False, "group": WC2026_TEAM_TO_GROUP.get(team)}
        for team in WC2026_ALL_TEAMS
    }

    qualified: list[str] = []
    third_places: list[dict] = []
    for group_name, teams in WC2026_GROUPS.items():
        ranked = simulate_group(teams, sampler, rng)
        group_stats[ranked[0][0]]["top_group"] = True
        for team, stats in ranked:
            group_stats[team]["points"] = float(stats["points"])
        qualified += [ranked[0][0], ranked[1][0]]
        third_places.append({"team": ranked[2][0], **ranked[2][1]})

    third_places.sort(key=lambda r: (r["points"], r["gd"], r["gf"], rng.random()), reverse=True)
    qualified += [r["team"] for r in third_places[:8]]

    # base-repo simplification kept for this mode: random knockout draw
    current = list(qualified)
    for round_name in KO_ROUNDS:
        stage = _ROUND_TO_STAGE[round_name]
        for team in current:
            stage_reached[team] = stage
        if len(current) <= 1:
            break
        rng.shuffle(current)
        current = [
            sampler.sample_knockout(current[i], current[i + 1], rng, round_name)[0]
            for i in range(0, len(current) - 1, 2)
        ]

    champion = current[0]
    stage_reached[champion] = "champion"
    return stage_reached, group_stats


def _empty_counters() -> dict:
    return {
        "stage": {team: {s: 0 for s in STAGES} for team in WC2026_ALL_TEAMS},
        "top_group": {team: 0 for team in WC2026_ALL_TEAMS},
        "points": {team: 0.0 for team in WC2026_ALL_TEAMS},
    }


_STAGE_ORDER = {s: i for i, s in enumerate(STAGES)}


def _accumulate(counters: dict, stage_reached: dict, group_stats: dict) -> None:
    for team, stage in stage_reached.items():
        team = to_wc_team(team)
        # reaching a stage implies having reached all earlier ones
        for s in STAGES[: _STAGE_ORDER[stage] + 1]:
            counters["stage"][team][s] += 1
    for team, stats in group_stats.items():
        counters["points"][team] += stats["points"]
        if stats["top_group"]:
            counters["top_group"][team] += 1


def _run_batch(n_sims: int, seed: int) -> dict:
    sampler = make_sampler()
    rng = np.random.default_rng(seed)
    counters = _empty_counters()
    for _ in range(n_sims):
        stage_reached, group_stats = simulate_one_tournament(sampler, rng)
        _accumulate(counters, stage_reached, group_stats)
    return counters


def _merge(target: dict, source: dict) -> None:
    for team in WC2026_ALL_TEAMS:
        for s in STAGES:
            target["stage"][team][s] += source["stage"][team][s]
        target["top_group"][team] += source["top_group"][team]
        target["points"][team] += source["points"][team]


def _split_batches(n_simulations: int, n_workers: int, seed: int) -> list[tuple[int, int]]:
    base, remainder = divmod(n_simulations, n_workers)
    return [
        (base + (1 if w < remainder else 0), seed + w * 1_000_003)
        for w in range(n_workers)
        if base + (1 if w < remainder else 0) > 0
    ]


def _default_worker_count() -> int:
    return max(1, min(os.cpu_count() or 1, 4))


def counters_to_dataframe(counters: dict, n_simulations: int) -> pd.DataFrame:
    rows = []
    for team in WC2026_ALL_TEAMS:
        st = counters["stage"][team]
        rows.append({
            "team": team,
            "group": WC2026_TEAM_TO_GROUP.get(team),
            "p_r32": st["r32"] / n_simulations,
            "p_r16": st["r16"] / n_simulations,
            "p_qf": st["qf"] / n_simulations,
            "p_sf": st["sf"] / n_simulations,
            "p_final": st["final"] / n_simulations,
            "p_champion": st["champion"] / n_simulations,
            "p_top_group": counters["top_group"][team] / n_simulations,
            "expected_points": counters["points"][team] / n_simulations,
            "simulations": n_simulations,
        })
    return (
        pd.DataFrame(rows)
        .sort_values(["p_champion", "p_final", "p_sf"], ascending=False)
        .reset_index(drop=True)
    )


def run_full_simulation(
    n_simulations: int = N_SIMULATIONS_DEFAULT,
    seed: int = 42,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Full-tournament Monte Carlo (mode a). Writes outputs/simulation_full.csv."""
    workers = max(1, min(n_workers or _default_worker_count(), n_simulations))
    print(f"Running {n_simulations} full-tournament simulations ({workers} workers)...")
    t0 = time.perf_counter()

    counters = _empty_counters()
    if workers == 1:
        counters = _run_batch(n_simulations, seed)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_batch, count, batch_seed): count
                for count, batch_seed in _split_batches(n_simulations, workers, seed)
            }
            done = 0
            for future in as_completed(futures):
                _merge(counters, future.result())
                done += futures[future]
                print(f"  ... {done}/{n_simulations}")

    elapsed = time.perf_counter() - t0
    print(f"  Finished in {elapsed:.1f}s ({n_simulations / elapsed:.0f} sims/s)")

    df = counters_to_dataframe(counters, n_simulations)
    save_path = OUTPUTS_DIR / "simulation_full.csv"
    df.to_csv(save_path, index=False)
    print(f"  [OK] Full simulation -> {save_path}")
    top = df.head(5)[["team", "p_champion"]]
    print("  Top 5:", [(r.team, f"{r.p_champion:.1%}") for r in top.itertuples()])
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WC2026 full-tournament Monte Carlo")
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS_DEFAULT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_full_simulation(args.simulations, seed=args.seed, n_workers=args.workers)
