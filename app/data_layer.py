import logging
from functools import lru_cache
from typing import Iterable, Optional

import pandas as pd

from .config import PROJECT_ROOT, require_file

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DATA_PATH = DATA_DIR / "assets.parquet"

_COLUMN_RENAMES = {
    "property_name": "address",
    "profit": "pnl",
    "entity_name": "entity",
    "ledger_type": "type",
    "ledger_group": "group",
    "ledger_category": "category",
    "ledger_code": "code",
    "ledger_description": "description",
}


def _normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {src: dst for src, dst in _COLUMN_RENAMES.items() if src in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    if "year" in df.columns and "period" not in df.columns:
        df["period"] = df["year"].astype(str)
    return df


@lru_cache(maxsize=1)
def _read_all_assets() -> pd.DataFrame:
    """
    Internal cached reader for the full dataset.
    """

    path = require_file(ASSETS_DATA_PATH)
    logger.info("Reading full assets dataset from %s", path)
    raw = pd.read_parquet(path)
    return _normalize_dataset(raw)


def load_assets(refresh: bool = False, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Load the assets dataset with optional caching for the full payload.
    """

    if refresh:
        logger.info("Clearing cached assets dataframe")
        _read_all_assets.cache_clear()

    df = _read_all_assets().copy()
    if columns:
        column_list = [col for col in columns if col in df.columns]
        missing = [col for col in columns if col not in df.columns]
        if missing:
            logger.warning("Requested columns %s missing from dataset", missing)
        if column_list:
            df = df[column_list]
        else:
            return pd.DataFrame()

    logger.info("Loaded %s asset records", len(df))
    return df


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

