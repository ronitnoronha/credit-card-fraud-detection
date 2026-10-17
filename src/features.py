"""Feature engineering.

Two groups of features:

1. Transaction-level (only need the row itself): hour, day_of_week, is_weekend,
   age at transaction time.
2. Card-history ("behavioural") features: computed per card using ONLY that
   card's earlier transactions. They never use the fraud label, because in
   production a label is only known weeks after the transaction (chargeback).

Why train and test are processed together: test starts strictly after train ends,
so a test transaction's history legitimately includes the card's train-period
transactions — exactly what a bank would know at scoring time. Because the
features look only backwards and never at labels, no test information can leak
into train rows.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.data_loader import load_splits

logger = logging.getLogger(__name__)

WINDOWS_SECONDS: dict[str, int] = {"card_txn_count_1h": 3600, "card_txn_count_24h": 86400}
MIN_STD_FOR_ZSCORE: float = 1.0  # dollars; avoids exploding z-scores for cards with near-constant spend


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[config.TIME_COL])
    df["ts"] = ts
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek  # Monday = 0
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = ts.dt.to_period("M").astype(str)  # EDA / SQL only, not a model feature
    df["age"] = ((ts - pd.to_datetime(df["dob"])).dt.days / 365.25).round(1)
    return df


def _window_counts(times: np.ndarray, window: int) -> np.ndarray:
    """Number of *earlier* transactions within `window` seconds (times sorted)."""
    idx = np.arange(len(times))
    return idx - np.searchsorted(times, times - window, side="left")


def add_card_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking per-card features. Requires `ts` from add_transaction_features."""
    df = df.copy()
    df["_order"] = np.arange(len(df))
    df = df.sort_values([config.CARD_COL, "ts", "_order"], kind="mergesort").reset_index(drop=True)

    card = df[config.CARD_COL]
    g = df.groupby(card, sort=False)
    # Explicit unit conversion: pandas may store datetimes at ns, us or s resolution.
    t_sec = df["ts"].astype("datetime64[s]").astype("int64")

    prior = g.cumcount()
    df["card_prior_txn_count"] = prior
    df["secs_since_prev_txn"] = t_sec.groupby(card, sort=False).diff()

    for name, window in WINDOWS_SECONDS.items():
        out = np.empty(len(df), dtype=np.int64)
        for _, positions in g.indices.items():
            out[positions] = _window_counts(t_sec.to_numpy()[positions], window)
        df[name] = out

    # Expanding mean/std of the card's PREVIOUS amounts (current row excluded).
    amt = df["amt"]
    prev_sum = amt.groupby(card, sort=False).cumsum() - amt
    prev_sq = (amt**2).groupby(card, sort=False).cumsum() - amt**2
    n = prior.astype(float)
    prev_mean = prev_sum / n.where(n > 0)
    prev_var = (prev_sq - n * prev_mean**2) / (n - 1).where(n > 1)
    prev_std = np.sqrt(prev_var.clip(lower=0)).clip(lower=MIN_STD_FOR_ZSCORE)
    df["amt_to_card_mean_ratio"] = amt / prev_mean
    df["amt_zscore_card"] = (amt - prev_mean) / prev_std

    # Share of the card's previous transactions that were in this same category.
    same_cat_prior = df.groupby([config.CARD_COL, "category"], sort=False).cumcount()
    df["card_category_share"] = same_cat_prior / n.where(n > 0)

    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def build_feature_table(use_cache: bool = True) -> pd.DataFrame:
    """Return train+test with engineered features and a `split` column."""
    path = config.PROCESSED_DATA_DIR / "features.parquet"
    if use_cache and path.exists():
        logger.info("Loading cached features from %s", path)
        return pd.read_parquet(path)

    splits = load_splits()
    frames = [splits[name].assign(split=name) for name in ("train", "test")]
    df = pd.concat(frames, ignore_index=True)
    logger.info("Building transaction features for %d rows", len(df))
    df = add_transaction_features(df)
    logger.info("Building card-history features")
    df = add_card_history_features(df)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Cached feature table to %s", path)
    return df


def features_from_card_summary(*, amt: float, category: str, hour: int, day_of_week: int, age: float,
                               city_pop: int, prior_txn_count: int, txn_count_1h: int, txn_count_24h: int,
                               secs_since_prev_txn: float | None, prior_mean_amt: float | None,
                               prior_std_amt: float | None, prior_same_category_count: int) -> pd.DataFrame:
    """Build one model-ready row from a card's history summary (used by the app).

    Mirrors add_card_history_features exactly, so a manually entered transaction is
    scored with the same feature definitions as training.
    """
    n = prior_txn_count
    mean = prior_mean_amt if n > 0 else None
    std = max(prior_std_amt, MIN_STD_FOR_ZSCORE) if (n > 1 and prior_std_amt is not None) else None
    row = {
        "amt": amt, "age": age, "city_pop": city_pop, "is_weekend": int(day_of_week >= 5),
        "card_txn_count_1h": txn_count_1h, "card_txn_count_24h": txn_count_24h,
        "secs_since_prev_txn": secs_since_prev_txn if n > 0 else np.nan,
        "amt_to_card_mean_ratio": amt / mean if mean else np.nan,
        "amt_zscore_card": (amt - mean) / std if (mean is not None and std) else np.nan,
        "card_category_share": prior_same_category_count / n if n > 0 else np.nan,
        "card_prior_txn_count": n, "hour": hour, "day_of_week": day_of_week, "category": category,
    }
    return pd.DataFrame([row])[model_features()]


def model_features(include_cold_start: bool = True) -> list[str]:
    feats = config.NUMERIC_FEATURES + config.CYCLIC_CATEGORICAL_FEATURES + config.CATEGORICAL_FEATURES
    if not include_cold_start:
        feats = [f for f in feats if f not in config.COLD_START_FEATURES]
    overlap = set(feats) & set(config.DROP_COLUMNS)
    assert not overlap, f"Dropped columns used as features: {overlap}"
    return feats


def get_xy(df: pd.DataFrame, split: str, include_cold_start: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    part = df[df["split"] == split]
    return part[model_features(include_cold_start)], part[config.TARGET].astype(int)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    table = build_feature_table(use_cache=False)
    print(table[model_features()].describe().T)
