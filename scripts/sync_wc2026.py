# Syncs WC2026 fixtures/results into SQLite with a source cascade so the
# project never blocks on the API:
#   1. API-Football (if APIFOOTBALL_KEY is set and covers season 2026)
#   2. martj42 results.csv (local copy; pass refresh=True / --refresh to
#      force a fresh download - it ships played AND upcoming WC2026 fixtures)
#   3. data/manual/wc2026_results.csv (hand-editable, applied as overrides)

"""Sync WC2026 matches into data/database.db. Run: python -m scripts.sync_wc2026"""

import argparse

import pandas as pd

from src.config import (
    MANUAL_DIR,
    RAW_DIR,
    WC2026_ALL_TEAMS,
    WC2026_ROUND_WINDOWS,
    WC2026_START_DATE,
    WC2026_TEAM_TO_GROUP,
)
from src.data import api_football, db
from src.utils.labels import normalize_team_name

MANUAL_CSV = MANUAL_DIR / "wc2026_results.csv"


def infer_round(date: str) -> str:
    for round_name, (start, end) in WC2026_ROUND_WINDOWS.items():
        if start <= date <= end:
            return round_name
    raise ValueError(f"date {date} does not fall in any WC2026 round window")


def _validate_teams(rows: list[dict], source: str) -> None:
    known = set(WC2026_ALL_TEAMS)
    unknown = sorted(
        {t for r in rows for t in (r["home_team"], r["away_team"])} - known
    )
    if unknown:
        raise ValueError(
            f"[{source}] teams not in WC2026_ALL_TEAMS (add TEAM_ALIASES entries): {unknown}"
        )


def rows_from_martj42() -> list[dict]:
    """WC2026 rows from the martj42 dataset (incl. scheduled NaN-score fixtures)."""
    path = RAW_DIR / "results.csv"
    if not path.exists():
        raise FileNotFoundError("data/raw/results.csv missing - run the historical download first")
    results = pd.read_csv(path)
    wc = results[
        (results["tournament"] == "FIFA World Cup")
        & (results["date"] >= WC2026_START_DATE)
    ].copy()
    wc["home_team"] = wc["home_team"].apply(normalize_team_name)
    wc["away_team"] = wc["away_team"].apply(normalize_team_name)

    # penalty winners live in a separate martj42 file
    pen_winner: dict[tuple, str] = {}
    pen_path = RAW_DIR / "shootouts.csv"
    if pen_path.exists():
        pens = pd.read_csv(pen_path)
        pens = pens[pens["date"] >= WC2026_START_DATE]
        for _, p in pens.iterrows():
            key = (p["date"], normalize_team_name(p["home_team"]), normalize_team_name(p["away_team"]))
            pen_winner[key] = normalize_team_name(p["winner"])

    rows = []
    for _, m in wc.iterrows():
        date = str(m["date"])[:10]
        played = pd.notna(m["home_score"]) and pd.notna(m["away_score"])
        round_name = infer_round(date)
        winner = pen_winner.get((date, m["home_team"], m["away_team"]))

        group_name = None
        if round_name == "group":
            gh = WC2026_TEAM_TO_GROUP.get(m["home_team"])
            ga = WC2026_TEAM_TO_GROUP.get(m["away_team"])
            group_name = gh if gh == ga else None

        rows.append({
            "date": date,
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "round": round_name,
            "stage": "group" if round_name == "group" else "knockout",
            "group_name": group_name,
            "home_goals": int(m["home_score"]) if played else None,
            "away_goals": int(m["away_score"]) if played else None,
            "penalty_winner": winner,
            "status": ("PEN" if winner else "FT") if played else "NS",
        })
    return rows


def sync_from_api() -> int | None:
    """Try the API. Returns rows written, or None if unavailable/uncovered."""
    if not api_football.get_api_key():
        print("  [skip] API-Football: no APIFOOTBALL_KEY in .env")
        return None
    try:
        payload = api_football.fetch_wc2026_fixtures()
        rows = api_football.fixtures_to_rows(payload)
    except Exception as e:
        print(f"  [skip] API-Football: {e}")
        return None
    if not rows:
        print("  [skip] API-Football returned 0 fixtures for season 2026 (free tier?)")
        return None
    _validate_teams(rows, "api")
    written = db.upsert_matches(rows, source="api")
    db.log_sync("api", written, f"{len(rows)} fixtures from API-Football")
    print(f"  [OK] API-Football: {written} matches upserted")
    return written


def sync_from_martj42() -> int:
    rows = rows_from_martj42()
    _validate_teams(rows, "martj42")
    written = db.upsert_matches(rows, source="martj42")
    played = sum(1 for r in rows if r["status"] != "NS")
    db.log_sync("martj42", written, f"{played} played / {len(rows)} total")
    print(f"  [OK] martj42 fallback: {written} matches upserted ({played} played)")
    return written


def apply_manual_overrides() -> int:
    if not MANUAL_CSV.exists():
        return 0
    manual = pd.read_csv(MANUAL_CSV)
    manual = manual.where(pd.notna(manual), None)
    rows = manual.to_dict("records")
    for r in rows:
        r["home_team"] = normalize_team_name(r["home_team"])
        r["away_team"] = normalize_team_name(r["away_team"])
        for col in ("home_goals", "away_goals"):
            if r.get(col) is not None:
                r[col] = int(r[col])
    _validate_teams(rows, "manual")
    written = db.upsert_matches(rows, source="manual")
    if written:
        db.log_sync("manual", written, str(MANUAL_CSV))
        print(f"  [OK] manual overrides: {written} matches upserted")
    return written


def print_summary() -> None:
    df = db.load_matches()
    played = df[df["status"] != "NS"]
    print("\nDatabase summary:")
    print(f"  total matches: {len(df)}  (played {len(played)}, scheduled {len(df) - len(played)})")
    for round_name in WC2026_ROUND_WINDOWS:
        sub = df[df["round"] == round_name]
        if len(sub):
            done = (sub["status"] != "NS").sum()
            print(f"    {round_name:13s} {done}/{len(sub)} played")


def main(source: str = "auto", refresh: bool = False) -> None:
    print("Syncing WC2026 matches -> data/database.db")
    db.init_db()
    if refresh:
        from src.data.historical_data import download_martj42_dataset
        print("  Refreshing martj42 dataset (forced download)...")
        download_martj42_dataset(force=True)
    if source in ("auto", "api"):
        written = sync_from_api()
        if written is None and source == "auto":
            sync_from_martj42()
        elif written is None:
            raise SystemExit("API sync requested but unavailable")
    elif source == "martj42":
        sync_from_martj42()
    apply_manual_overrides()
    print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync WC2026 matches into SQLite")
    parser.add_argument(
        "--source", choices=["auto", "api", "martj42"], default="auto",
        help="auto: API if available, else martj42 (manual CSV always applied last)",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="force re-download of the martj42 dataset before syncing",
    )
    args = parser.parse_args()
    main(source=args.source, refresh=args.refresh)
