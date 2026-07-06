"""Manual WC2026 results: export template / import overrides.

  python -m scripts.import_manual_results --template   # DB -> data/manual/wc2026_results.csv
  python -m scripts.import_manual_results              # CSV -> DB (source=manual)
"""

import argparse

from src.config import MANUAL_DIR
from src.data import db
from scripts.sync_wc2026 import MANUAL_CSV, apply_manual_overrides

TEMPLATE_COLS = [
    "date", "home_team", "away_team", "round", "stage", "group_name",
    "home_goals", "away_goals", "penalty_winner", "status",
]


def export_template() -> None:
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    df = db.load_matches()
    if df.empty:
        print("[WARN] database is empty - run scripts.sync_wc2026 first; writing header only")
    df = df.reindex(columns=TEMPLATE_COLS)
    df.to_csv(MANUAL_CSV, index=False)
    print(f"[OK] template with {len(df)} matches -> {MANUAL_CSV}")
    print("     Edit scores/status by hand (status: NS, FT, AET or PEN), then run")
    print("     python -m scripts.import_manual_results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual WC2026 results import/export")
    parser.add_argument("--template", action="store_true", help="export DB -> editable CSV")
    args = parser.parse_args()
    if args.template:
        export_template()
    else:
        written = apply_manual_overrides()
        if not written:
            print(f"[WARN] nothing imported - is {MANUAL_CSV} present and non-empty?")
