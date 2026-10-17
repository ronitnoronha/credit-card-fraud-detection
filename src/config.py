"""Central project configuration.

All paths are resolved relative to the project root so nothing depends on a
machine-specific absolute path. Every modelling choice that someone might want
to change (feature lists, CV folds, threshold objective, risk tiers) lives here.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
SHAP_FIGURES_DIR: Path = FIGURES_DIR / "shap"
DASHBOARD_DATA_DIR: Path = REPORTS_DIR / "dashboard"
SQL_DIR: Path = PROJECT_ROOT / "sql"

# Hugging Face dataset id (override with env var if a mirror is used).
HF_DATASET_ID: str = os.getenv("HF_DATASET_ID", "pointe77/credit-card-transaction")

RANDOM_SEED: int = 42

# ---------------------------------------------------------------- columns
TARGET: str = "is_fraud"
TIME_COL: str = "trans_date_trans_time"
CARD_COL: str = "cc_num"  # grouping key only — never a model feature

# Columns dropped before modelling (see reports/data_leakage_analysis.md).
DROP_COLUMNS: list[str] = [
    "Unnamed: 0", "trans_num", "unix_time",           # row index / IDs / redundant time
    "first", "last", "street", "city", "zip", "job",   # PII & card-identity proxies
    "merchant", "state", "lat", "long",                # high-cardinality / identity proxies
    "merch_lat", "merch_long", "merch_zipcode",        # synthetic jitter, no signal
    "gender",                                          # excluded by design decision (fairness)
    "dob",                                             # replaced by `age`
]

# ---------------------------------------------------------------- features
CATEGORICAL_FEATURES: list[str] = ["category"]
# Temporal features: numeric for trees; one-hot encoded for Logistic Regression
# because fraud risk is not monotonic in the hour of day.
CYCLIC_CATEGORICAL_FEATURES: list[str] = ["hour", "day_of_week"]
NUMERIC_FEATURES: list[str] = [
    "amt",
    "age",
    "city_pop",
    "is_weekend",
    "card_txn_count_1h",
    "card_txn_count_24h",
    "secs_since_prev_txn",
    "amt_to_card_mean_ratio",
    "amt_zscore_card",
    "card_category_share",
    "card_prior_txn_count",
]
# Heavily right-skewed features: log1p-transformed for Logistic Regression only
# (trees are invariant to monotonic transforms).
SKEWED_FEATURES: list[str] = [
    "amt", "city_pop", "secs_since_prev_txn", "card_prior_txn_count",
    "card_txn_count_1h", "card_txn_count_24h", "amt_to_card_mean_ratio",
]
# Feature that can exploit the "fraud-only new card" artefact of this simulator.
COLD_START_FEATURES: list[str] = ["card_prior_txn_count"]

# ---------------------------------------------------------------- modelling
CV_FOLDS: int = 5
# Model comparison / tuning runs on a stratified random subsample of train to keep
# runtime reasonable on a laptop. The final model is refit on the full train set.
CV_SAMPLE_FRACTION: float = float(os.getenv("CV_SAMPLE_FRACTION", "0.3"))
N_JOBS: int = int(os.getenv("N_JOBS", "-1"))
TUNING_ITERATIONS: int = int(os.getenv("TUNING_ITERATIONS", "20"))

# ---------------------------------------------------------------- threshold
# Business objective (fixed BEFORE looking at results): maximise recall subject to
# alert precision >= MIN_PRECISION, i.e. at most one false alert per fraud caught,
# which keeps a fraud-analyst review queue manageable.
MIN_PRECISION: float = 0.50

# ---------------------------------------------------------------- risk tiers
# Low    : score < alert threshold (the MIN_PRECISION rule above) -> approve.
# Medium : alert threshold <= score < high threshold -> step-up authentication /
#          analyst review. Precision in this band is lower, so add friction rather than decline.
# High   : score >= threshold where out-of-fold precision reaches HIGH_TIER_MIN_PRECISION
#          (at most ~1 false decline per 9 frauds) -> decline / hold automatically.
# NOTE: the original medium rule ("threshold reaching 95% recall") was replaced after the
# first run showed it fell *above* the alert threshold (the alert rule already reached
# 96.6% recall), which made the tiers inconsistent. The 0.90 floor is a business choice
# made after seeing the precision-recall curve; it is documented as such.
HIGH_TIER_MIN_PRECISION: float = 0.90

# ---------------------------------------------------------------- plotting
COLOR_LEGIT: str = "#2a78d6"
COLOR_FRAUD: str = "#eb6834"
COLOR_TEXT: str = "#0b0b0b"
COLOR_TEXT_MUTED: str = "#52514e"
COLOR_GRID: str = "#e4e3df"
COLOR_SURFACE: str = "#fcfcfb"
