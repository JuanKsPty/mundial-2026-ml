# "Simulate from the current state" mode: reads the real WC2026 results from
# SQLite plus the official knockout adjacency (data/manual/bracket_2026.json)
# and Monte Carlo-simulates ONLY the remaining matches, fixing everything
# already played.

"""Current-state tournament snapshot + Monte Carlo of the remaining bracket.

Run: python -m src.simulation.tournament_state [--simulations 5000]
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import MANUAL_DIR, N_SIMULATIONS_DEFAULT, OUTPUTS_DIR
from src.data import db
from src.simulation.montecarlo import MatchSampler, make_sampler

BRACKET_PATH = MANUAL_DIR / "bracket_2026.json"

_ROUND_STAGE = {
    "round_of_16": "qf",      # winning your R16 match means reaching the QF
    "quarterfinal": "sf",
    "semifinal": "final",
    "final": "champion",
}


@dataclass
class KOSlot:
    slot_id: str                    # e.g. "r16_3", "qf_1", "final"
    round: str                      # round_of_16 | quarterfinal | semifinal | final
    date: str | None = None
    home: str | None = None         # resolved team, if known
    away: str | None = None
    feeds: tuple[str, str] | None = None  # slot ids whose winners meet here
    played: bool = False
    winner: str | None = None
    score: tuple[int, int] | None = None


@dataclass
class TournamentSnapshot:
    slots: dict[str, KOSlot]
    order: list[str]                # resolution order (rounds in sequence)
    alive: list[str]                # teams not yet eliminated
    n_pending: int


def _match_db_row(df: pd.DataFrame, home: str, away: str, round_name: str):
    """Find the DB row for a pairing (order-insensitive) in a given round."""
    sel = df[
        (df["round"] == round_name)
        & (
            ((df["home_team"] == home) & (df["away_team"] == away))
            | ((df["home_team"] == away) & (df["away_team"] == home))
        )
    ]
    return sel.iloc[0] if len(sel) else None


def _row_winner(row) -> str | None:
    if row is None or row["status"] == "NS":
        return None
    if row["penalty_winner"]:
        return row["penalty_winner"]
    if row["home_goals"] > row["away_goals"]:
        return row["home_team"]
    if row["away_goals"] > row["home_goals"]:
        return row["away_team"]
    return None  # draw without recorded shootout: treat as pending


def build_snapshot() -> TournamentSnapshot:
    """Combine the official bracket adjacency with real results from SQLite."""
    if not BRACKET_PATH.exists():
        raise FileNotFoundError(f"{BRACKET_PATH} missing (official KO adjacency)")
    with open(BRACKET_PATH, encoding="utf-8") as f:
        bracket = json.load(f)

    matches = db.load_matches(stage="knockout")
    if matches.empty:
        raise FileNotFoundError("no knockout matches in DB - run scripts.sync_wc2026")

    slots: dict[str, KOSlot] = {}
    order: list[str] = []

    for slot_def in bracket["slots"]:
        slot = KOSlot(
            slot_id=slot_def["id"],
            round=slot_def["round"],
            date=slot_def.get("date"),
            home=slot_def.get("home"),
            away=slot_def.get("away"),
            feeds=tuple(slot_def["feeds"]) if slot_def.get("feeds") else None,
        )
        slots[slot.slot_id] = slot
        order.append(slot.slot_id)

    # resolve feeds using real winners, then attach played results
    for slot_id in order:
        slot = slots[slot_id]
        if slot.feeds:
            w1 = slots[slot.feeds[0]].winner
            w2 = slots[slot.feeds[1]].winner
            slot.home = slot.home or w1
            slot.away = slot.away or w2
        if slot.home and slot.away:
            row = _match_db_row(matches, slot.home, slot.away, slot.round)
            winner = _row_winner(row)
            if winner is not None:
                slot.played = True
                slot.winner = winner
                slot.score = (int(row["home_goals"]), int(row["away_goals"]))

    r16_teams = [t for s in slots.values() if s.round == "round_of_16"
                 for t in (s.home, s.away) if t]
    if len(r16_teams) != 16:
        raise ValueError(f"expected 16 R16 teams, got {len(r16_teams)} - check bracket/DB")
    eliminated_r16 = {
        t for s in slots.values() if s.round == "round_of_16" and s.played
        for t in (s.home, s.away) if t != s.winner
    }
    alive = [t for t in r16_teams if t not in eliminated_r16]
    n_pending = sum(1 for s in slots.values() if not s.played)
    return TournamentSnapshot(slots=slots, order=order, alive=alive, n_pending=n_pending)


def simulate_from_current_state(
    n_simulations: int = N_SIMULATIONS_DEFAULT,
    seed: int = 42,
    sampler: MatchSampler | None = None,
) -> pd.DataFrame:
    """Monte Carlo of the remaining bracket (mode b). Played matches are fixed;
    pending ones are sampled. Writes outputs/simulation_current.csv."""
    snapshot = build_snapshot()
    sampler = sampler or make_sampler()
    rng = np.random.default_rng(seed)

    r16_teams = [t for s in snapshot.slots.values() if s.round == "round_of_16"
                 for t in (s.home, s.away)]
    stages = ["qf", "sf", "final", "champion"]
    counts = {t: dict.fromkeys(stages, 0) for t in r16_teams}
    # per-slot occupancy/winner counts, for the bracket UI
    slot_side: dict[str, tuple[dict, dict]] = {
        sid: ({}, {}) for sid in snapshot.order
    }
    slot_wins: dict[str, dict] = {sid: {} for sid in snapshot.order}

    print(f"Simulating {n_simulations}x the remaining bracket "
          f"({snapshot.n_pending} pending matches, {len(snapshot.alive)} teams alive)...")

    for _ in range(n_simulations):
        winners: dict[str, str] = {}
        for slot_id in snapshot.order:
            slot = snapshot.slots[slot_id]
            home = slot.home or winners[slot.feeds[0]]
            away = slot.away or winners[slot.feeds[1]]
            if slot.played:
                w = slot.winner
            else:
                w = sampler.sample_knockout(home, away, rng, slot.round)[0]
            winners[slot_id] = w
            counts[w][_ROUND_STAGE[slot.round]] += 1
            h_side, a_side = slot_side[slot_id]
            h_side[home] = h_side.get(home, 0) + 1
            a_side[away] = a_side.get(away, 0) + 1
            slot_wins[slot_id][w] = slot_wins[slot_id].get(w, 0) + 1

    _write_bracket_probabilities(snapshot, slot_side, slot_wins, n_simulations)

    rows = []
    for team in r16_teams:
        c = counts[team]
        rows.append({
            "team": team,
            "alive": team in snapshot.alive,
            "p_qf": c["qf"] / n_simulations,
            "p_sf": c["sf"] / n_simulations,
            "p_final": c["final"] / n_simulations,
            "p_champion": c["champion"] / n_simulations,
            "simulations": n_simulations,
        })
    df = (
        pd.DataFrame(rows)
        .sort_values(["p_champion", "p_final", "p_sf", "p_qf"], ascending=False)
        .reset_index(drop=True)
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUTS_DIR / "simulation_current.csv"
    df.to_csv(save_path, index=False)
    print(f"  [OK] Current-state simulation -> {save_path}")
    top = df.head(5)[["team", "p_champion"]]
    print("  Top 5:", [(r.team, f"{r.p_champion:.1%}") for r in top.itertuples()])
    return df


def _write_bracket_probabilities(
    snapshot: TournamentSnapshot,
    slot_side: dict[str, tuple[dict, dict]],
    slot_wins: dict[str, dict],
    n_simulations: int,
) -> pd.DataFrame:
    """Per-slot occupancy and win probabilities -> outputs/bracket_probabilities.csv.
    One row per (slot, side, candidate team): p_present = prob the team plays
    that slot on that side; p_win_match = prob it wins that slot (overall)."""
    rows = []
    for slot_id in snapshot.order:
        slot = snapshot.slots[slot_id]
        for side, side_counts in zip(("home", "away"), slot_side[slot_id]):
            for team, c in sorted(side_counts.items(), key=lambda kv: -kv[1]):
                rows.append({
                    "slot": slot_id,
                    "round": slot.round,
                    "date": slot.date,
                    "played": slot.played,
                    "score": f"{slot.score[0]}-{slot.score[1]}" if slot.score else "",
                    "winner": slot.winner or "",
                    "side": side,
                    "team": team,
                    "p_present": c / n_simulations,
                    "p_win_match": slot_wins[slot_id].get(team, 0) / n_simulations,
                })
    df = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / "bracket_probabilities.csv"
    df.to_csv(path, index=False)
    print(f"  [OK] Bracket probabilities -> {path}")
    return df


def pending_bracket_summary() -> pd.DataFrame:
    """Human-readable view of the bracket state (used by the dashboard)."""
    snapshot = build_snapshot()
    rows = []
    for slot_id in snapshot.order:
        s = snapshot.slots[slot_id]
        rows.append({
            "slot": slot_id,
            "round": s.round,
            "home": s.home or f"winner {s.feeds[0]}",
            "away": s.away or f"winner {s.feeds[1]}",
            "played": s.played,
            "score": f"{s.score[0]}-{s.score[1]}" if s.score else "",
            "winner": s.winner or "",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate the remaining WC2026 bracket")
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(pending_bracket_summary().to_string(index=False))
    simulate_from_current_state(args.simulations, seed=args.seed)
