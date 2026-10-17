"""Phase 9 — SHAP explanations for the final XGBoost pipeline.

TreeExplainer computes exact SHAP values for tree ensembles. For one transaction:

    model log-odds = base value + sum(SHAP value of each feature)

so each SHAP value is that feature's additive contribution (in log-odds) to pushing
this score above or below the average score. This describes how the MODEL uses
features; it is not a causal statement about why fraud happened.

Usage: python -m src.explainability
"""
from __future__ import annotations

import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config
from src.features import build_feature_table, model_features
from src.plotting import apply_style, save
from src.risk import assign_tier, load_thresholds

logger = logging.getLogger(__name__)
OUT = config.SHAP_FIGURES_DIR
SAMPLE_SIZE = 20_000


def load_model():
    return joblib.load(config.MODELS_DIR / "xgb_pipeline.joblib")


def get_explainer(pipeline) -> shap.TreeExplainer:
    return shap.TreeExplainer(pipeline.named_steps["model"])


def explain_rows(pipeline, explainer: shap.TreeExplainer, X_raw: pd.DataFrame) -> shap.Explanation:
    """SHAP values for raw (untransformed) rows; displayed data = the raw values."""
    X_t = pipeline.named_steps["prep"].transform(X_raw)
    sv = explainer(X_t)
    # Show human-readable raw values (e.g. category name instead of its ordinal code).
    sv.data = X_raw[X_t.columns].to_numpy(dtype=object)
    return sv


def contribution_table(sv_row: shap.Explanation) -> pd.DataFrame:
    df = pd.DataFrame({"feature": sv_row.feature_names, "value": sv_row.data, "shap_log_odds": sv_row.values})
    df["direction"] = np.where(df["shap_log_odds"] > 0, "raises risk", "lowers risk")
    return df.reindex(df["shap_log_odds"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def plot_waterfall(sv_row: shap.Explanation, title: str, path=None, max_display: int = 12):
    plt.figure()
    shap.plots.waterfall(sv_row, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 5.5)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, fontweight="bold")
    if path is not None:
        save(fig, path)
    return fig


def run() -> None:
    apply_style()
    pipeline = load_model()
    explainer = get_explainer(pipeline)
    thr = load_thresholds()
    feats = model_features()

    df = build_feature_table()
    test = df[df["split"] == "test"].reset_index(drop=True)
    sample = test.sample(n=SAMPLE_SIZE, random_state=config.RANDOM_SEED)
    sv = explain_rows(pipeline, explainer, sample[feats])
    X_t = pipeline.named_steps["prep"].transform(sample[feats])

    # 1) Global: XGBoost's own gain importance (how much each feature improved the
    #    training loss across splits) — model-internal, no direction.
    booster = pipeline.named_steps["model"].get_booster()
    gain = pd.Series(booster.get_score(importance_type="gain")).reindex(X_t.columns).fillna(0)
    gain = (gain / gain.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(gain.index, gain.values, color=config.COLOR_LEGIT, height=0.6)
    ax.set(title="XGBoost gain importance (share of total gain)", xlabel="Share of total gain")
    ax.grid(axis="y", visible=False)
    save(fig, OUT / "xgb_gain_importance.png")

    # 2) Global SHAP importance: mean |SHAP| on a random test sample.
    imp = pd.Series(np.abs(sv.values).mean(axis=0), index=X_t.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color=config.COLOR_FRAUD, height=0.6)
    ax.set(title=f"SHAP feature importance (mean |SHAP|, {SAMPLE_SIZE:,} test txns)", xlabel="Mean |SHAP value| (log-odds)")
    ax.grid(axis="y", visible=False)
    save(fig, OUT / "shap_importance.png")
    pd.DataFrame({"mean_abs_shap": imp[::-1], "gain_share": gain.reindex(imp.index[::-1])}).to_csv(
        config.REPORTS_DIR / "feature_importance.csv")

    # 3) Beeswarm summary (needs numeric data for colouring -> use transformed values).
    sv_num = shap.Explanation(sv.values, base_values=sv.base_values, data=X_t.to_numpy(), feature_names=list(X_t.columns))
    plt.figure()
    shap.plots.beeswarm(sv_num, max_display=len(X_t.columns), show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    fig.suptitle("SHAP summary: each dot is one test transaction", x=0.02, ha="left", fontsize=11, fontweight="bold")
    save(fig, OUT / "shap_summary.png")

    # 4) Individual explanations: a caught fraud (true positive) and a false alert (false positive).
    proba = pipeline.predict_proba(test[feats])[:, 1]
    test = test.assign(proba=proba, tier=assign_tier(proba, thr))
    flagged = test["proba"] >= thr.medium  # operating (alert) threshold
    tp = test[flagged & (test[config.TARGET] == 1)]
    fp = test[flagged & (test[config.TARGET] == 0)]
    examples = {}
    # Median-scored example of each group, so it is typical rather than the most extreme.
    for label, group in [("true_positive", tp), ("false_positive", fp)]:
        if group.empty:
            continue
        row = group.iloc[[int(np.argsort(group["proba"].to_numpy())[len(group) // 2])]]
        sv_row = explain_rows(pipeline, explainer, row[feats])[0]
        title = (f"{label.replace('_', ' ').title()}: P(fraud)={row['proba'].iloc[0]:.3f}, tier={row['tier'].iloc[0]}, "
                 f"actual={'fraud' if row[config.TARGET].iloc[0] else 'legitimate'}")
        plot_waterfall(sv_row, title, OUT / f"waterfall_{label}.png")
        table = contribution_table(sv_row)
        table.to_csv(config.REPORTS_DIR / f"shap_example_{label}.csv", index=False)
        examples[label] = (row, table)
        logger.info("%s\n%s", title, table.head(8).to_string(index=False))
    logger.info("SHAP figures written to %s", OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
