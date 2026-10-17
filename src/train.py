"""Model training, comparison, tuning, threshold selection and final evaluation.

Stages (run all with `python -m src.train`, or one with `--stage <name>`):

  compare  Phase 5+6: compare class-imbalance strategies for each model with
           5-fold StratifiedGroupKFold (grouped by card) on a stratified subsample
           of train; write reports/imbalance_strategy_comparison.csv and
           reports/model_comparison.csv.
  tune     Phase 7: RandomizedSearchCV for XGBoost and Random Forest (scored on
           PR-AUC), then re-evaluate the tuned configs with the same 5-fold CV.
  final    Phase 8: out-of-fold predictions on the FULL train set to choose the
           decision threshold and risk tiers; refit models on full train;
           evaluate once on the out-of-time test set; ablations; save artefacts.

The test set is touched only in `final`, after every choice has been made on train.
"""
from __future__ import annotations

import argparse
import json
import os
import logging
import time
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.evaluate import (
    compute_metrics,
    plot_confusion,
    plot_pr_roc,
    plot_threshold_tradeoff,
    threshold_for_min_precision,
    threshold_max_f1,
    threshold_table,
)
from src.features import build_feature_table, model_features
from src.plotting import apply_style
from src.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)

ARTIFACTS = config.MODELS_DIR / "training_state.json"
RF_TUNING_ITERATIONS = int(os.getenv("RF_TUNING_ITERATIONS", "6"))
SMOTE_RATIO = 0.1  # oversample minority to 10% of majority (not 50/50) to limit synthetic noise

MODEL_KIND = {"Logistic Regression": "linear", "Random Forest": "tree", "XGBoost": "tree"}
STRATEGIES = {
    "Logistic Regression": ["none", "class_weight", "smote"],
    "Random Forest": ["none", "class_weight"],  # SMOTE skipped: doubles an already slow fit, interpolates category codes
    "XGBoost": ["none", "scale_pos_weight", "smote"],
}


# ------------------------------------------------------------------ model factory
def base_estimator(name: str, strategy: str, pos_weight: float, params: dict | None = None):
    params = dict(params or {})
    seed = config.RANDOM_SEED
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, C=params.pop("C", 1.0),
                                  class_weight="balanced" if strategy == "class_weight" else None,
                                  random_state=seed)
    if name == "Random Forest":
        defaults = dict(n_estimators=200, max_depth=16, min_samples_leaf=5, max_features="sqrt")
        defaults.update(params)
        if strategy == "class_weight":
            defaults.setdefault("class_weight", "balanced_subsample")
        return RandomForestClassifier(n_jobs=config.N_JOBS, random_state=seed, **defaults)
    if name == "XGBoost":
        defaults = dict(n_estimators=400, learning_rate=0.1, max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0)
        defaults.update(params)
        if strategy == "scale_pos_weight":
            defaults.setdefault("scale_pos_weight", pos_weight)
        return XGBClassifier(tree_method="hist", eval_metric="aucpr", n_jobs=config.N_JOBS,
                             random_state=seed, **defaults)
    raise ValueError(name)


def make_pipeline(name: str, strategy: str, features: list[str], pos_weight: float, params: dict | None = None):
    """Preprocessing + (optional SMOTE) + model in one object, so every fit is leak-free."""
    pre = build_preprocessor(MODEL_KIND[name], features)
    model = base_estimator(name, strategy, pos_weight, params)
    if strategy == "smote":
        # imblearn's Pipeline applies SMOTE during fit only; predict/transform skip it,
        # so validation and test data are never resampled.
        return ImbPipeline([("prep", pre), ("smote", SMOTE(sampling_strategy=SMOTE_RATIO, random_state=config.RANDOM_SEED)),
                            ("model", model)])
    return Pipeline([("prep", pre), ("model", model)])


# ------------------------------------------------------------------ CV
def grouped_cv(n_splits: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)


def cross_validate_oof(factory: Callable[[], object], X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                       n_splits: int = config.CV_FOLDS, label: str = "") -> tuple[pd.DataFrame, np.ndarray]:
    """Return per-fold metrics (at threshold 0.5) and out-of-fold probabilities."""
    oof = np.zeros(len(y))
    rows = []
    for fold, (tr, va) in enumerate(grouped_cv(n_splits).split(X, y, groups)):
        t0 = time.time()
        model = factory()
        model.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        m = compute_metrics(y.iloc[va], oof[va]).as_dict()
        rows.append({"fold": fold, **m})
        logger.info("[%s] fold %d  PR-AUC=%.4f ROC-AUC=%.4f  (%.0fs)", label, fold, m["pr_auc"], m["roc_auc"], time.time() - t0)
    return pd.DataFrame(rows), oof


def summarise(fold_df: pd.DataFrame, oof: np.ndarray, y: pd.Series, model: str, strategy: str) -> dict:
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    out = {"model": model, "strategy": strategy}
    for c in cols:
        out[c] = fold_df[c].mean()
        out[f"{c}_std"] = fold_df[c].std()
    agg = compute_metrics(y, oof)
    out.update({"oof_tn": agg.tn, "oof_fp": agg.fp, "oof_fn": agg.fn, "oof_tp": agg.tp})
    return out


# ------------------------------------------------------------------ data
def load_train_test(include_cold_start: bool = True):
    df = build_feature_table()
    feats = model_features(include_cold_start)
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, test, feats


def cv_sample(train: pd.DataFrame) -> pd.DataFrame:
    """Stratified random subsample (keeps the natural fraud rate)."""
    s = train.groupby(config.TARGET, group_keys=False).sample(frac=config.CV_SAMPLE_FRACTION,
                                                               random_state=config.RANDOM_SEED)
    return s.reset_index(drop=True)


def pos_weight_of(y: pd.Series) -> float:
    return float((y == 0).sum() / (y == 1).sum())


def _load_state() -> dict:
    return json.loads(ARTIFACTS.read_text()) if ARTIFACTS.exists() else {}


def _save_state(state: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.write_text(json.dumps(state, indent=2, default=float))


# ------------------------------------------------------------------ stages
def stage_compare() -> None:
    train, _, feats = load_train_test()
    sample = cv_sample(train)
    X, y, groups = sample[feats], sample[config.TARGET].astype(int), sample[config.CARD_COL]
    pw = pos_weight_of(y)
    logger.info("CV sample: %d rows, %d frauds (%.3f%%), %d cards; scale_pos_weight=%.1f",
                len(y), y.sum(), 100 * y.mean(), groups.nunique(), pw)

    results, oofs = [], {}
    for name, strategies in STRATEGIES.items():
        for strat in strategies:
            fold_df, oof = cross_validate_oof(lambda: make_pipeline(name, strat, feats, pw), X, y, groups,
                                              label=f"{name}/{strat}")
            results.append(summarise(fold_df, oof, y, name, strat))
            oofs[(name, strat)] = oof

    res = pd.DataFrame(results)
    res.to_csv(config.REPORTS_DIR / "imbalance_strategy_comparison.csv", index=False)

    best = res.loc[res.groupby("model")["pr_auc"].idxmax()]
    comp = best[["model", "strategy", "precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy",
                 "pr_auc_std", "oof_tn", "oof_fp", "oof_fn", "oof_tp"]].copy()
    comp.insert(2, "evaluation", "5-fold StratifiedGroupKFold (CV sample), threshold 0.5")
    comp.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)

    apply_style()
    curves = {r.model: (y.to_numpy(), oofs[(r.model, r.strategy)]) for r in best.itertuples()}
    plot_pr_roc(curves, "(out-of-fold, CV sample)", config.FIGURES_DIR / "model_cv_pr_roc.png")

    state = _load_state()
    state["best_strategy"] = dict(zip(best["model"], best["strategy"]))
    _save_state(state)
    logger.info("\n%s", comp.round(4).to_string(index=False))


def stage_tune() -> None:
    state = _load_state()
    train, _, feats = load_train_test()
    sample = cv_sample(train)
    X, y, groups = sample[feats], sample[config.TARGET].astype(int), sample[config.CARD_COL]
    pw = pos_weight_of(y)

    spaces = {
        "XGBoost": (config.TUNING_ITERATIONS, {
            "model__n_estimators": randint(200, 801),
            "model__learning_rate": loguniform(0.02, 0.3),
            "model__max_depth": randint(3, 11),
            "model__min_child_weight": randint(1, 11),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_lambda": loguniform(0.1, 10),
            "model__gamma": uniform(0, 5),
            "model__scale_pos_weight": [1.0, float(np.sqrt(pw)), pw],
        }),
        "Random Forest": (RF_TUNING_ITERATIONS, {
            "model__n_estimators": [200, 300],
            "model__max_depth": [12, 16, 24],
            "model__min_samples_leaf": [2, 5, 10, 20],
            "model__max_features": ["sqrt", 0.5],
            "model__class_weight": [None, "balanced_subsample"],
        }),
    }
    tuned = {}
    for name, (n_iter, space) in spaces.items():
        logger.info("Tuning %s: %d candidates x 3 folds", name, n_iter)
        t0 = time.time()
        search = RandomizedSearchCV(make_pipeline(name, "none", feats, pw), space, n_iter=n_iter,
                                    scoring="average_precision", cv=grouped_cv(3), refit=False,
                                    random_state=config.RANDOM_SEED, n_jobs=1, verbose=0)
        search.fit(X, y, groups=groups)
        cvres = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        cvres.to_csv(config.REPORTS_DIR / f"tuning_{name.lower().replace(' ', '_')}.csv", index=False)
        params = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
        tuned[name] = params
        logger.info("%s best 3-fold PR-AUC=%.4f in %.0fs: %s", name, search.best_score_, time.time() - t0, params)

    # Re-evaluate tuned configs with the same 5-fold CV used for the untuned comparison.
    comp = pd.read_csv(config.REPORTS_DIR / "model_comparison.csv")
    comp = comp[~comp["model"].str.contains("tuned")]
    rows = []
    for name, params in tuned.items():
        fold_df, oof = cross_validate_oof(lambda: make_pipeline(name, "none", feats, pw, params), X, y, groups,
                                          label=f"{name}/tuned")
        s = summarise(fold_df, oof, y, f"{name} (tuned)", "tuned")
        s["evaluation"] = "5-fold StratifiedGroupKFold (CV sample), threshold 0.5"
        rows.append(s)
    comp = pd.concat([comp, pd.DataFrame(rows)[comp.columns]], ignore_index=True)
    comp.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    logger.info("\n%s", comp.round(4).to_string(index=False))

    state["tuned_params"] = tuned
    _save_state(state)


def stage_final() -> None:
    state = _load_state()
    tuned = state["tuned_params"]
    train, test, feats = load_train_test()
    X, y, groups = train[feats], train[config.TARGET].astype(int), train[config.CARD_COL]
    X_te, y_te = test[feats], test[config.TARGET].astype(int)
    pw = pos_weight_of(y)
    apply_style()

    # ---- 1. Threshold selection on out-of-fold predictions over the FULL train set.
    xgb_factory = lambda: make_pipeline("XGBoost", "none", feats, pw, tuned["XGBoost"])  # noqa: E731
    fold_df, oof = cross_validate_oof(xgb_factory, X, y, groups, label="XGBoost/full-train OOF")
    t_alert = threshold_for_min_precision(y, oof, config.MIN_PRECISION)
    t_high = threshold_for_min_precision(y, oof, config.HIGH_TIER_MIN_PRECISION)
    t_f1 = threshold_max_f1(y, oof)
    grid = np.unique(np.round(np.concatenate([np.arange(0.05, 1.0, 0.05), [t_alert, t_high, t_f1, 0.5]]), 4))
    ttab = threshold_table(y, oof, grid)
    ttab["note"] = ""
    ttab.loc[np.isclose(ttab["threshold"], round(t_alert, 4)), "note"] += "chosen (max recall s.t. precision>=%.2f) " % config.MIN_PRECISION
    ttab.loc[np.isclose(ttab["threshold"], round(t_high, 4)), "note"] += "high-tier floor (precision>=%.2f) " % config.HIGH_TIER_MIN_PRECISION
    ttab.loc[np.isclose(ttab["threshold"], round(t_f1, 4)), "note"] += "max F1 "
    ttab.loc[np.isclose(ttab["threshold"], 0.5), "note"] += "default 0.5"
    ttab.to_csv(config.REPORTS_DIR / "threshold_analysis.csv", index=False)
    plot_threshold_tradeoff(threshold_table(y, oof, np.round(np.arange(0.01, 1.0, 0.01), 2)), t_alert,
                            config.FIGURES_DIR / "threshold_tradeoff.png")
    oof_metrics = {"at_0.5": compute_metrics(y, oof, 0.5).as_dict(),
                   "at_chosen": compute_metrics(y, oof, t_alert).as_dict(),
                   "fold_pr_auc_mean": fold_df["pr_auc"].mean(), "fold_pr_auc_std": fold_df["pr_auc"].std()}
    logger.info("Thresholds: alert/medium=%.4f high=%.4f maxF1=%.4f", t_alert, t_high, t_f1)

    # ---- 2. Refit each model on full train, evaluate once on the out-of-time test set.
    specs = {
        "Logistic Regression": (state["best_strategy"]["Logistic Regression"], None),
        "Random Forest": ("none", tuned["Random Forest"]),
        "XGBoost": ("none", tuned["XGBoost"]),
    }
    test_rows, curves, fitted = [], {}, {}
    for name, (strategy, params) in specs.items():
        t0 = time.time()
        model = make_pipeline(name, strategy, feats, pw, params).fit(X, y)
        p = model.predict_proba(X_te)[:, 1]
        fitted[name], curves[name] = model, (y_te.to_numpy(), p)
        for label, thr in [("0.5", 0.5)] + ([("chosen", t_alert)] if name == "XGBoost" else []):
            m = compute_metrics(y_te, p, thr).as_dict()
            test_rows.append({"model": name, "strategy": strategy if params is None else "tuned",
                              "threshold_rule": label, **m})
        logger.info("%s test PR-AUC=%.4f (fit %.0fs)", name, compute_metrics(y_te, p).pr_auc, time.time() - t0)
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(config.REPORTS_DIR / "test_metrics.csv", index=False)
    plot_pr_roc(curves, "(out-of-time test set)", config.FIGURES_DIR / "model_test_pr_roc.png")

    p_xgb = curves["XGBoost"][1]
    m_chosen = compute_metrics(y_te, p_xgb, t_alert)
    plot_confusion(m_chosen, f"XGBoost — test set, threshold {t_alert:.3f}", config.FIGURES_DIR / "confusion_matrix_xgb_test.png")
    plot_confusion(compute_metrics(y_te, p_xgb, 0.5), "XGBoost — test set, threshold 0.5",
                   config.FIGURES_DIR / "confusion_matrix_xgb_test_0.5.png")
    for name in ("Logistic Regression", "Random Forest"):
        plot_confusion(compute_metrics(y_te, curves[name][1], 0.5), f"{name} — test set, threshold 0.5",
                       config.FIGURES_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}_test.png")

    # ---- 3. Robustness checks for the "fraud-only new card" artefact.
    seen = test[config.CARD_COL].isin(set(train[config.CARD_COL]))
    ablation = [{"check": "XGBoost, all test cards", **_threshold_free(y_te, p_xgb),
                 **_at(y_te, p_xgb, t_alert)},
                {"check": "XGBoost, test excluding cards unseen in train", **_threshold_free(y_te[seen], p_xgb[seen]),
                 **_at(y_te[seen], p_xgb[seen], t_alert), "n_rows_removed": int((~seen).sum())}]
    feats_nc = model_features(include_cold_start=False)
    model_nc = make_pipeline("XGBoost", "none", feats_nc, pw, tuned["XGBoost"]).fit(train[feats_nc], y)
    p_nc = model_nc.predict_proba(test[feats_nc])[:, 1]
    # Threshold-free metrics only: the alert threshold was tuned for the full model, so
    # applying it to a different model would not be a like-for-like comparison.
    ablation.append({"check": "XGBoost without card_prior_txn_count, all test cards",
                     **_threshold_free(y_te, p_nc)})
    pd.DataFrame(ablation).to_csv(config.REPORTS_DIR / "robustness_checks.csv", index=False)

    # ---- 4. Persist artefacts used by explainability + the Streamlit app.
    joblib.dump(fitted["XGBoost"], config.MODELS_DIR / "xgb_pipeline.joblib")
    joblib.dump(fitted["Logistic Regression"], config.MODELS_DIR / "logreg_pipeline.joblib")
    metadata = {
        "features": feats,
        "tuned_params": tuned,
        "thresholds": {"alert": t_alert, "medium": t_alert, "high": t_high, "max_f1_reference": t_f1,
                       "objective": f"max recall s.t. precision >= {config.MIN_PRECISION} on out-of-fold train predictions",
                       "high_rule": f"lowest threshold with precision >= {config.HIGH_TIER_MIN_PRECISION} on out-of-fold train predictions"},
        "oof_train_metrics": oof_metrics,
        "test_metrics_xgb_chosen": m_chosen.as_dict(),
        "test_metrics_xgb_0.5": compute_metrics(y_te, p_xgb, 0.5).as_dict(),
        "train_period": [str(train["ts"].min()), str(train["ts"].max())],
        "test_period": [str(test["ts"].min()), str(test["ts"].max())],
    }
    (config.MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float))
    # Small, shareable copy for reports / the app (models/ is git-ignored).
    (config.REPORTS_DIR / "final_model_summary.json").write_text(json.dumps(metadata, indent=2, default=float))
    logger.info("\n%s", test_df.round(4).to_string(index=False))


def _threshold_free(y, p) -> dict:
    m = compute_metrics(y, p)
    return {"pr_auc": m.pr_auc, "roc_auc": m.roc_auc}


def _at(y, p, thr) -> dict:
    m = compute_metrics(y, p, thr)
    return {"threshold": thr, "precision": m.precision, "recall": m.recall, "f1": m.f1, "fp": m.fp, "fn": m.fn}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["compare", "tune", "final", "all"], default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stages = {"compare": stage_compare, "tune": stage_tune, "final": stage_final}
    for name in (stages if args.stage == "all" else [args.stage]):
        logger.info("=== stage: %s ===", name)
        stages[name]()


if __name__ == "__main__":
    main()
