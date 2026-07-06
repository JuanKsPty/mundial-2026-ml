"""Validate API-Football free-tier coverage of WC2026 (uses <= 3 requests).

Run: python -m scripts.validate_api
Verdicts:
  FULL_OK      - season 2026 fixtures available; API can be the primary source
  NOT_COVERED  - key works but season 2026 is not accessible on this plan
  NO_KEY       - no APIFOOTBALL_KEY configured; use the martj42 fallback
"""

from src.config import WC2026_ALL_TEAMS, WC_SEASON
from src.data import api_football


def main() -> str:
    if not api_football.get_api_key():
        print("[NO_KEY] APIFOOTBALL_KEY not set in .env - the sync will use the")
        print("         martj42 fallback (already covers played WC2026 matches).")
        return "NO_KEY"

    # 1) account status (does not consume daily quota)
    status = api_football.fetch_status()["response"]
    account, sub, req = status.get("account", {}), status.get("subscription", {}), status.get("requests", {})
    print(f"Account : {account.get('firstname', '?')} ({account.get('email', '?')})")
    print(f"Plan    : {sub.get('plan', '?')} (active={sub.get('active')})")
    print(f"Quota   : {req.get('current', '?')}/{req.get('limit_day', '?')} requests today")

    # 2) league coverage (1 request)
    leagues = api_football.fetch_league_seasons()["response"]
    seasons = [s["year"] for lg in leagues for s in lg.get("seasons", [])]
    print(f"WorldCup seasons visible on this plan: {sorted(seasons)}")
    if WC_SEASON not in seasons:
        print(f"[NOT_COVERED] season {WC_SEASON} not listed - use the martj42 fallback.")
        return "NOT_COVERED"

    # 3) fixtures (1 request). The league listing may show 2026 even when the
    #    plan can't query it, so the real test is the fixtures call itself.
    try:
        payload = api_football.fetch_wc2026_fixtures()
    except RuntimeError as e:
        print(f"[NOT_COVERED] {e}")
        print("              -> use the martj42 fallback (scripts.sync_wc2026)")
        return "NOT_COVERED"
    rows = api_football.fixtures_to_rows(payload)
    if not rows:
        print(f"[NOT_COVERED] 0 fixtures returned for season {WC_SEASON}.")
        return "NOT_COVERED"

    played = sum(1 for r in rows if r["status"] != "NS")
    print(f"Fixtures: {len(rows)} total, {played} played")
    unmapped = sorted(
        {t for r in rows for t in (r["home_team"], r["away_team"])}
        - set(WC2026_ALL_TEAMS)
    )
    if unmapped:
        print(f"[WARN] team names needing TEAM_ALIASES entries: {unmapped}")
    print("[FULL_OK] API-Football can be the primary WC2026 source.")
    return "FULL_OK"


if __name__ == "__main__":
    main()
