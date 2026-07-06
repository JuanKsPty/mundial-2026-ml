"""Central configuration: paths, data sources, WC2026 draw and model constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MANUAL_DIR = ROOT_DIR / "data" / "manual"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DB_PATH = ROOT_DIR / "data" / "database.db"

# ---------------------------------------------------------------------------
# Historical data sources
# ---------------------------------------------------------------------------
WC_YEARS = [1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]

MARTJ42_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
MARTJ42_FILES = {
    "results": "results.csv",
    "goalscorers": "goalscorers.csv",
    "shootouts": "shootouts.csv",
    "former_names": "former_names.csv",
}

FIFA_RANKINGS_API = "https://api.fifa.com/api/v3/rankings"
FIFA_RANKINGS_LIMIT = 300

MATCH_MODEL_START_YEAR = 2000
CACHE_TTL_DAYS = 7  # martj42 / FIFA refresh window

# ---------------------------------------------------------------------------
# Temporal split for training / calibration / evaluation
# A chronological split keeps the calibration and test metrics honest
# (no peeking into the future).
# ---------------------------------------------------------------------------
TRAIN_END_YEAR = 2021    # train:       year <= 2021
CALIB_START_YEAR = 2022  # calibration: 2022-2024
TEST_START_YEAR = 2025   # test:        2025 -> TRAIN_MAX_DATE
TRAIN_MAX_DATE = "2026-06-10"  # hard cutoff: nothing from the ongoing WC2026

# ---------------------------------------------------------------------------
# API-Football (api-sports.io) - WC2026 fixtures/results
# ---------------------------------------------------------------------------
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_KEY_ENV = "APIFOOTBALL_KEY"
API_FOOTBALL_RATE_LIMIT_SEC = 6  # free tier: 10 req/min
WC_LEAGUE_ID = 1     # FIFA World Cup
WC_SEASON = 2026

WC2026_START_DATE = "2026-06-11"
WC2026_END_DATE = "2026-07-19"

# Date windows to infer the round when the source (martj42/manual) lacks it.
# (start, end) inclusive, from the official FIFA calendar.
WC2026_ROUND_WINDOWS = {
    "group": ("2026-06-11", "2026-06-27"),
    "round_of_32": ("2026-06-28", "2026-07-03"),
    "round_of_16": ("2026-07-04", "2026-07-07"),
    "quarterfinal": ("2026-07-09", "2026-07-11"),
    "semifinal": ("2026-07-14", "2026-07-15"),
    "third_place": ("2026-07-18", "2026-07-18"),
    "final": ("2026-07-19", "2026-07-19"),
}

# ---------------------------------------------------------------------------
# Team name normalization
# ---------------------------------------------------------------------------
TEAM_ALIASES = {
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea, South": "South Korea",
    "USA": "United States",
    "United States": "United States",
    "United States of America": "United States",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Czech Republic": "Czechia",
    "Turkey": "Türkiye",
    "Turkiye": "Türkiye",
    "DR Congo": "Congo DR",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Democratic Republic of the Congo": "Congo DR",
    "Chinese Taipei": "Taiwan",
    "FYR Macedonia": "North Macedonia",
    "Macedonia": "North Macedonia",
    "SUI": "Switzerland",
    "KSA": "Saudi Arabia",
    "CIV": "Ivory Coast",
    "MAR": "Morocco",
    "RSA": "South Africa",
    "IRL": "Republic of Ireland",
    "COD": "Congo DR",
    "CPV": "Cape Verde",
    "CUW": "Curaçao",
    "NCL": "New Caledonia",
    "SUR": "Suriname",
}

# ---------------------------------------------------------------------------
# WC2026 draw (48 teams, 12 groups - official)
# ---------------------------------------------------------------------------
WC2026_GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "Congo DR"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

WC2026_ALL_TEAMS = sorted({
    team for teams in WC2026_GROUPS.values() for team in teams
})

WC2026_TEAM_TO_GROUP = {
    team: group for group, teams in WC2026_GROUPS.items() for team in teams
}

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
# Per-round unpredictability, expressed as sigma of a lognormal noise applied
# to the Poisson goal rates.
ROUND_GOAL_NOISE = {
    "group": 0.08,
    "round_of_32": 0.12,
    "round_of_16": 0.14,
    "quarterfinal": 0.16,
    "semifinal": 0.18,
    "third_place": 0.18,
    "final": 0.20,
}

# Recency-weighted tactical/squad quality (2022-2026 cycle).
MODERN_TEAMS = {
    "France": 6, "Argentina": 6, "Spain": 6, "England": 5, "Brazil": 5,
    "Portugal": 5, "Germany": 5, "Morocco": 4, "Netherlands": 4, "Croatia": 4,
    "Colombia": 3, "Japan": 3, "Senegal": 3, "Uruguay": 3, "United States": 3,
    "Belgium": 1, "Switzerland": 1, "Mexico": 1, "Italy": 4,
}

N_SIMULATIONS_DEFAULT = 5000
MAX_GOALS = 6  # score matrix covers 0..MAX_GOALS goals per team
