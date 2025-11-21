import logging
from functools import lru_cache
from typing import Iterable, Optional

import pandas as pd

from .config import PROJECT_ROOT, require_file

logger = logging.getLogger(__name__)

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
def _read_all_assets() -> pd.DataFrame:
    """
    Internal cached reader for the full dataset.
    """

    path = require_file(ASSETS_DATA_PATH)
    logger.info("Reading full assets dataset from %s", path)
    return pd.read_parquet(path)


def _read_assets(columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Read the dataset, optionally selecting a subset of columns (uncached).
    """

    path = require_file(ASSETS_DATA_PATH)
    column_list = list(columns) if columns else None
    if column_list:
        logger.info("Reading asset columns %s from %s", column_list, path)
    return pd.read_parquet(path, columns=column_list)


def load_assets(refresh: bool = False, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Load the assets dataset with optional caching for the full payload.
    """

    if refresh:
        logger.info("Clearing cached assets dataframe")
        _read_all_assets.cache_clear()

    if columns:
        df = _read_assets(columns=columns)
    else:
        df = _read_all_assets()

    logger.info("Loaded %s asset records", len(df))
    return df.copy()


def summarize_assets(df: pd.DataFrame) -> dict:
    """
    Return high level stats for diagnostics.
    """

    summary = {"rows": int(len(df)), "columns": list(df.columns)}
    numeric = df.select_dtypes("number")
    if not numeric.empty:
        summary["numerics"] = numeric.describe().to_dict()
    logger.info("Assets summary generated: %s", summary)
    return summary


__all__ = ["load_assets", "summarize_assets", "ASSETS_DATA_PATH"]

