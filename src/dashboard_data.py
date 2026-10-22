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


def threshold_curve(y_true: np.ndarray, proba: np.ndarray, amt: np.ndarray, extra: list[float]) -> pd.DataFrame:
    """Confusion counts and fraud dollars caught/missed over a log-spaced threshold grid."""
    grid = np.unique(np.concatenate([np.round(np.logspace(-3, np.log10(0.999), 300), 6), extra]))
    order = np.argsort(-proba)
    p_sorted, y_sorted, amt_sorted = proba[order], y_true[order], amt[order]
    tp_cum = np.concatenate([[0], np.cumsum(y_sorted)])
    fraud_amt_cum = np.concatenate([[0], np.cumsum(amt_sorted * y_sorted)])
    # Number of transactions with proba >= t (proba sorted descending).
    n_flagged = np.searchsorted(-p_sorted, -grid, side="right")
    tp, fraud_caught = tp_cum[n_flagged], fraud_amt_cum[n_flagged]
    fp = n_flagged - tp
    fn = y_true.sum() - tp
    out = pd.DataFrame({"threshold": grid, "tp": tp, "fp": fp, "fn": fn, "tn": len(y_true) - y_true.sum() - fp,
                        "fraud_usd_caught": fraud_caught, "fraud_usd_missed": fraud_amt_cum[-1] - fraud_caught})
    out["precision"] = out["tp"] / (out["tp"] + out["fp"]).replace(0, np.nan)
    out["recall"] = out["tp"] / y_true.sum()
    out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"])
    return out


def run() -> None:
    thr = load_thresholds()
    model = joblib.load(config.MODELS_DIR / "xgb_pipeline.joblib")
    full = build_feature_table()
    test = full[full["split"] == "test"].copy()
    test["proba"] = model.predict_proba(test[model_features()])[:, 1]
    config.DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    curve = threshold_curve(test[config.TARGET].to_numpy(), test["proba"].to_numpy(), test["amt"].to_numpy(),
                            extra=[thr.medium, thr.high, 0.5])
    curve.to_csv(config.DASHBOARD_DATA_DIR / "threshold_curve_test.csv", index=False)
    logger.info("Wrote threshold curve with %d thresholds", len(curve))

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
