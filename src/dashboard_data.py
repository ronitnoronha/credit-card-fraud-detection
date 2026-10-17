"""Prepare the small files the Streamlit app needs (no raw data, no PII).

Writes reports/dashboard/example_transactions.csv: a sample of scored test
transactions (true/false positives and negatives) with the card-history summary
inputs the app's predictor form expects.

Usage: python -m src.dashboard_data   (after src.train and src.run_sql)
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import build_feature_table, model_features
from src.risk import assign_tier, load_thresholds

logger = logging.getLogger(__name__)
PER_GROUP = 40


def _history_summary(full: pd.DataFrame, row: pd.Series) -> dict:
    hist = full[(full[config.CARD_COL] == row[config.CARD_COL]) & (full["ts"] < row["ts"])]
    return {
        "prior_mean_amt": hist["amt"].mean() if len(hist) else np.nan,
        "prior_std_amt": hist["amt"].std() if len(hist) > 1 else np.nan,
        "prior_same_category_count": int((hist["category"] == row["category"]).sum()),
    }


def run() -> None:
    thr = load_thresholds()
    model = joblib.load(config.MODELS_DIR / "xgb_pipeline.joblib")
    full = build_feature_table()
    test = full[full["split"] == "test"].copy()
    test["proba"] = model.predict_proba(test[model_features()])[:, 1]
    flagged = test["proba"] >= thr.medium  # operating (alert) threshold
    y = test[config.TARGET] == 1
    groups = {"caught fraud (TP)": flagged & y, "false alert (FP)": flagged & ~y,
              "missed fraud (FN)": ~flagged & y, "legitimate (TN)": ~flagged & ~y}
    parts = []
    for label, mask in groups.items():
        g = test[mask]
        parts.append(g.sample(n=min(PER_GROUP, len(g)), random_state=config.RANDOM_SEED).assign(outcome=label))
    sample = pd.concat(parts)
    # Exact same-second ties are excluded from history here but counted by the training
    # features; the per-row card_prior_txn_count is kept from the feature table.
    summaries = pd.DataFrame([_history_summary(full, r) for _, r in sample.iterrows()], index=sample.index)
    out = pd.concat([sample[model_features() + [config.TARGET, "proba", "outcome"]], summaries], axis=1)
    out["tier"] = assign_tier(out["proba"], thr)
    config.DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.reset_index(drop=True).to_csv(config.DASHBOARD_DATA_DIR / "example_transactions.csv", index=False)
    logger.info("Wrote %d example transactions", len(out))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
