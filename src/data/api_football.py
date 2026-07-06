"""API-Football (api-sports.io) client for WC2026 fixtures and results."""

import os
import time

import requests
from dotenv import load_dotenv

from src.config import (
    API_FOOTBALL_BASE_URL,
    API_FOOTBALL_KEY_ENV,
    API_FOOTBALL_RATE_LIMIT_SEC,
    WC2026_TEAM_TO_GROUP,
    WC_LEAGUE_ID,
    WC_SEASON,
)
from src.utils.labels import normalize_team_name

# API round strings -> canonical round names used across the project
API_ROUND_MAP = {
    "round of 32": "round_of_32",
    "round of 16": "round_of_16",
    "quarter-finals": "quarterfinal",
    "quarterfinals": "quarterfinal",
    "semi-finals": "semifinal",
    "semifinals": "semifinal",
    "3rd place final": "third_place",
    "third place": "third_place",
    "final": "final",
}

_last_request_ts = 0.0


def get_api_key() -> str | None:
    load_dotenv()
    return os.environ.get(API_FOOTBALL_KEY_ENV) or os.environ.get("API_FOOTBALL_KEY")


def api_get(endpoint: str, params: dict | None = None, api_key: str | None = None) -> dict:
    """GET against api-sports.io v3 honoring the free-tier rate limit."""
    global _last_request_ts
    key = api_key or get_api_key()
    if not key:
        raise RuntimeError(f"No API key: set {API_FOOTBALL_KEY_ENV} in .env")

    wait = API_FOOTBALL_RATE_LIMIT_SEC - (time.time() - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.time()

    resp = requests.get(
        f"{API_FOOTBALL_BASE_URL}/{endpoint.lstrip('/')}",
        params=params or {},
        headers={"x-apisports-key": key},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        raise RuntimeError(f"API-Football error on /{endpoint}: {payload['errors']}")
    return payload


def fetch_status() -> dict:
    """Account status (plan, quota). Does not consume the daily quota."""
    return api_get("status")


def fetch_league_seasons() -> dict:
    """World Cup league entry, incl. covered seasons (1 request)."""
    return api_get("leagues", {"id": WC_LEAGUE_ID})


def fetch_wc2026_fixtures() -> dict:
    """All WC2026 fixtures - played and scheduled - in a single request."""
    return api_get("fixtures", {"league": WC_LEAGUE_ID, "season": WC_SEASON})


def _map_round(api_round: str) -> str:
    r = str(api_round).strip().lower()
    if r.startswith("group"):
        return "group"
    return API_ROUND_MAP.get(r, r.replace(" ", "_"))


def _map_status(short: str) -> str:
    # API short codes: NS/TBD (scheduled), FT (full time), AET, PEN
    if short in ("FT", "AET", "PEN"):
        return short
    return "NS"


def fixtures_to_rows(payload: dict) -> list[dict]:
    """Convert an API-Football fixtures payload to `matches` table rows."""
    rows = []
    for item in payload.get("response", []):
        fixture = item["fixture"]
        home = normalize_team_name(item["teams"]["home"]["name"])
        away = normalize_team_name(item["teams"]["away"]["name"])
        status = _map_status(fixture.get("status", {}).get("short", "NS"))
        round_name = _map_round(item.get("league", {}).get("round", ""))

        goals_home = item.get("goals", {}).get("home")
        goals_away = item.get("goals", {}).get("away")
        if status == "NS":
            goals_home = goals_away = None

        penalty_winner = None
        pen = item.get("score", {}).get("penalty", {})
        if status == "PEN" and pen.get("home") is not None:
            penalty_winner = home if pen["home"] > pen["away"] else away

        group_name = None
        if round_name == "group":
            gh, ga = WC2026_TEAM_TO_GROUP.get(home), WC2026_TEAM_TO_GROUP.get(away)
            group_name = gh if gh == ga else None

        rows.append({
            "date": str(fixture["date"])[:10],
            "home_team": home,
            "away_team": away,
            "round": round_name,
            "stage": "group" if round_name == "group" else "knockout",
            "group_name": group_name,
            "home_goals": goals_home,
            "away_goals": goals_away,
            "penalty_winner": penalty_winner,
            "status": status,
            "fixture_id": fixture.get("id"),
        })
    return rows
