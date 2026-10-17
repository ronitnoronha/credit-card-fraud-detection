"""Load the credit-card transaction dataset from Hugging Face.

The first call downloads the dataset through the `datasets` library and caches
each split as Parquet under data/raw/ (git-ignored). Later calls read the
Parquet cache, which is much faster than re-parsing the CSVs.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import HF_DATASET_ID, RAW_DATA_DIR

logger = logging.getLogger(__name__)


def _cache_path(split: str):
    return RAW_DATA_DIR / f"{split}.parquet"


def load_splits(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Return {split_name: DataFrame} for every split the dataset exposes."""
    cached = sorted(RAW_DATA_DIR.glob("*.parquet")) if use_cache else []
    if cached:
        logger.info("Loading cached splits from %s", RAW_DATA_DIR)
        return {p.stem: pd.read_parquet(p) for p in cached}

    # Fallback: CSVs placed manually in data/raw/ (e.g. train.csv, test.csv).
    local_csvs = sorted(RAW_DATA_DIR.glob("*.csv"))
    if local_csvs:
        logger.info("Loading local CSVs from %s", RAW_DATA_DIR)
        splits = {p.stem: pd.read_csv(p) for p in local_csvs}
        for name, df in splits.items():
            df.to_parquet(_cache_path(name), index=False)
        return splits

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the `datasets` package: pip install datasets") from exc

    logger.info("Downloading %s from Hugging Face", HF_DATASET_ID)
    try:
        ds = load_dataset(HF_DATASET_ID)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load '{HF_DATASET_ID}' from Hugging Face. Check network "
            "access to huggingface.co (and HF_TOKEN if the dataset is gated)."
        ) from exc

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    splits: dict[str, pd.DataFrame] = {}
    for name, split in ds.items():
        df = split.to_pandas()
        df.to_parquet(_cache_path(name), index=False)
        logger.info("Split %-6s -> %s rows x %s cols (cached)", name, *df.shape)
        splits[name] = df
    return splits
