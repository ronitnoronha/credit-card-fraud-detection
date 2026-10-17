"""Phase 10 — risk scoring layer: fraud probability -> Low / Medium / High.

Tier boundaries are NOT hand-picked numbers. They are derived from business rules
(src/config.py) applied to out-of-fold predictions on the training set:

  Medium : p >= alert threshold — max recall subject to precision >= MIN_PRECISION
           (0.50). Action: step-up authentication (one-time passcode) / analyst review.
  High   : p >= high threshold — lowest score where precision >= HIGH_TIER_MIN_PRECISION
           (0.90). Action: decline / hold automatically.
  Low    : otherwise — approve.

The derived numbers are stored in models/metadata.json by `src.train`. They can be
overridden with the env vars RISK_MEDIUM_THRESHOLD / RISK_HIGH_THRESHOLD.

Usage: python -m src.risk   (writes reports/risk_tier_summary.csv on the test set)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)
TIERS = ("Low", "Medium", "High")


@dataclass(frozen=True)
class RiskThresholds:
    medium: float
    high: float

    def __post_init__(self):
        if not 0 <= self.medium < self.high <= 1:
            raise ValueError(f"Need 0 <= medium < high <= 1, got {self.medium}, {self.high}")


def load_thresholds() -> RiskThresholds:
    meta_path = config.MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        meta_path = config.REPORTS_DIR / "final_model_summary.json"
    t = json.loads(meta_path.read_text())["thresholds"]
    return RiskThresholds(
        medium=float(os.getenv("RISK_MEDIUM_THRESHOLD", t["medium"])),
        high=float(os.getenv("RISK_HIGH_THRESHOLD", t["high"])),
    )


def assign_tier(proba, thr: RiskThresholds) -> np.ndarray:
    p = np.atleast_1d(np.asarray(proba, dtype=float))
    return np.select([p >= thr.high, p >= thr.medium], ["High", "Medium"], default="Low")


def tier_summary(y_true, proba, thr: RiskThresholds) -> pd.DataFrame:
    df = pd.DataFrame({"tier": assign_tier(proba, thr), "y": np.asarray(y_true)})
    s = df.groupby("tier")["y"].agg(n_transactions="size", n_fraud="sum")
    s = s.reindex(list(TIERS)).fillna(0).astype(int)
    s["fraud_rate_pct"] = 100 * s["n_fraud"] / s["n_transactions"].clip(lower=1)
    s["share_of_transactions_pct"] = 100 * s["n_transactions"] / s["n_transactions"].sum()
    s["share_of_all_fraud_pct"] = 100 * s["n_fraud"] / max(s["n_fraud"].sum(), 1)
    return s


def main() -> None:
    from src.features import build_feature_table, model_features

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    thr = load_thresholds()
    model = joblib.load(config.MODELS_DIR / "xgb_pipeline.joblib")
    df = build_feature_table()
    test = df[df["split"] == "test"]
    proba = model.predict_proba(test[model_features()])[:, 1]
    summary = tier_summary(test[config.TARGET], proba, thr)
    summary.to_csv(config.REPORTS_DIR / "risk_tier_summary.csv")
    logger.info("Thresholds: medium=%.4f high=%.4f\n%s", thr.medium, thr.high, summary.round(3).to_string())


if __name__ == "__main__":
    main()
