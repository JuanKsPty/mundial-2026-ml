"""Team name normalization shared by every data source."""

import pandas as pd

from src.config import TEAM_ALIASES


def normalize_team_name(name: str) -> str:
    if pd.isna(name):
        return name
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)
