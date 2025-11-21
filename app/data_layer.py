from functools import lru_cache
from typing import Iterable, Optional

import pandas as pd

from .config import PROJECT_ROOT, require_file


DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DATA_PATH = DATA_DIR / "assets.parquet"


# TODO: Create data type and Schema validation for the assets dataset
# @lru_cache(maxsize=1)
# def _read_assets(columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
#     """
#     Internal cached reader for the assets parquet file.
#     """
#
#     path = require_file(ASSETS_DATA_PATH)
#     return pd.read_parquet(path, columns=list(columns) if columns else None)

@lru_cache(maxsize=1)
def _read_assets() -> pd.DataFrame:
    if not ASSETS_DATA_PATH.exists():
        raise FileNotFoundError(f"Assets dataset not found at {ASSETS_DATA_PATH}")
    return pd.read_parquet(ASSETS_DATA_PATH)

def load_assets(refresh: bool = False) -> pd.DataFrame:
    if refresh:
        _read_assets.cache_clear()
    return _read_assets().copy()


# TODO

def load_assets(refresh: bool = False, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Load the assets dataset with optional caching.
    """

    if refresh:
        _read_assets.cache_clear()
    return _read_assets(columns=columns).copy()


def summarize_assets(df: pd.DataFrame) -> dict:
    """
    Return high level stats for diagnostics.
    """

    summary = {"rows": int(len(df)), "columns": list(df.columns)}
    numeric = df.select_dtypes("number")
    if not numeric.empty:
        summary["numerics"] = numeric.describe().to_dict()
    return summary


__all__ = ["load_assets", "summarize_assets", "ASSETS_DATA_PATH"]

