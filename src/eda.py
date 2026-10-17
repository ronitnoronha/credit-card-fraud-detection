"""Phase 1 — exploratory data analysis on the TRAIN split only.

The test set is kept out of EDA so that no modelling decision is influenced by it.
Each figure answers one question (see FIGURE_QUESTIONS); a markdown summary with
the underlying numbers is written to reports/eda_summary.md.

Usage: python -m src.eda
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.features import build_feature_table
from src.inspect_data import rank_auc
from src.plotting import apply_style, save

logger = logging.getLogger(__name__)
FIG = config.FIGURES_DIR

FIGURE_QUESTIONS: dict[str, str] = {
    "01_class_distribution.png": "How imbalanced is the target in train and test?",
    "02_amount_distribution.png": "How are transaction amounts distributed, and do fraud amounts differ from legitimate ones?",
    "03_fraud_rate_by_category.png": "Which merchant categories carry the highest fraud rate?",
    "04_fraud_rate_by_time.png": "Does fraud concentrate at particular hours or days?",
    "05_fraud_rate_by_state.png": "Is state a reliable geographic signal, or just noise from few cards per state?",
    "06_fraud_rate_by_age.png": "Does fraud rate vary with cardholder age?",
    "07_monthly_trend.png": "Is transaction volume and fraud rate stable over time?",
    "08_engineered_feature_separation.png": "Do the card-history features separate fraud from legitimate transactions?",
    "09_numeric_distributions.png": "How are the numeric inputs distributed for each class?",
}


def _rate_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    t = df.groupby(by, observed=True)[config.TARGET].agg(n_txn="size", n_fraud="sum", fraud_rate="mean")
    t["fraud_rate_pct"] = t["fraud_rate"] * 100
    return t


def plot_class_distribution(full: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, split in zip(axes, ["train", "test"]):
        counts = full.loc[full["split"] == split, config.TARGET].value_counts().sort_index()
        bars = ax.bar(["Legitimate", "Fraud"], counts.values, color=[config.COLOR_LEGIT, config.COLOR_FRAUD], width=0.55)
        ax.set_yscale("log")
        ax.set_title(f"{split.title()} split (log scale)")
        for b, v in zip(bars, counts.values):
            ax.annotate(f"{v:,}\n({v / counts.sum():.2%})", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=9, color=config.COLOR_TEXT)
        ax.set_ylim(top=counts.max() * 8)
        ax.grid(axis="x", visible=False)
    save(fig, FIG / "01_class_distribution.png")


def plot_amount(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.logspace(0, np.log10(train["amt"].max()), 60)
    for label, color, val in [("Legitimate", config.COLOR_LEGIT, 0), ("Fraud", config.COLOR_FRAUD, 1)]:
        a = train.loc[train[config.TARGET] == val, "amt"]
        axes[0].hist(a, bins=bins, density=True, histtype="step", color=color, linewidth=2, label=label)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Transaction amount ($, log scale)")
    axes[0].set_ylabel("Density (within class)")
    axes[0].set_title("Amount distribution by class")
    axes[0].legend()

    data = [train.loc[train[config.TARGET] == v, "amt"] for v in (0, 1)]
    bp = axes[1].boxplot(data, tick_labels=["Legitimate", "Fraud"], showfliers=False, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], [config.COLOR_LEGIT, config.COLOR_FRAUD]):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    for med in bp["medians"]:
        med.set_color(config.COLOR_TEXT)
    axes[1].set_ylabel("Amount ($)")
    axes[1].set_title("Amount by class (outliers hidden)")
    save(fig, FIG / "02_amount_distribution.png")


def plot_category(train: pd.DataFrame, overall: float) -> pd.DataFrame:
    t = _rate_table(train, "category").sort_values("fraud_rate_pct")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(t.index, t["fraud_rate_pct"], color=config.COLOR_FRAUD, height=0.6)
    ax.axvline(overall * 100, color=config.COLOR_TEXT_MUTED, linestyle="--", linewidth=1)
    ax.annotate(f"overall {overall:.2%}", (overall * 100, -0.9), color=config.COLOR_TEXT_MUTED, fontsize=8)
    for y, (rate, n) in enumerate(zip(t["fraud_rate_pct"], t["n_txn"])):
        ax.annotate(f" {rate:.2f}%  (n={n:,})", (rate, y), va="center", fontsize=8, color=config.COLOR_TEXT_MUTED)
    ax.set_xlabel("Fraud rate (%)")
    ax.set_title("Fraud rate by merchant category (train)")
    ax.set_xlim(right=t["fraud_rate_pct"].max() * 1.45)
    ax.grid(axis="y", visible=False)
    save(fig, FIG / "03_fraud_rate_by_category.png")
    return t


def plot_time(train: pd.DataFrame, overall: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    hour = _rate_table(train, "hour")
    dow = _rate_table(train, "day_of_week")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].plot(hour.index, hour["fraud_rate_pct"], color=config.COLOR_FRAUD, marker="o", markersize=4)
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Fraud rate (%)")
    axes[0].set_title("Fraud rate by hour")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1].bar([names[i] for i in dow.index], dow["fraud_rate_pct"], color=config.COLOR_FRAUD, width=0.6)
    axes[1].set_title("Fraud rate by weekday")
    axes[1].grid(axis="x", visible=False)
    for ax in axes:
        ax.axhline(overall * 100, color=config.COLOR_TEXT_MUTED, linestyle="--", linewidth=1)
    save(fig, FIG / "04_fraud_rate_by_time.png")
    return hour, dow


def plot_state(train: pd.DataFrame, overall: float) -> pd.DataFrame:
    t = _rate_table(train, "state")
    t["n_cards"] = train.groupby("state")[config.CARD_COL].nunique()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.scatter(t["n_cards"], t["fraud_rate_pct"], color=config.COLOR_FRAUD, s=36,
               edgecolor=config.COLOR_SURFACE, linewidth=1.5)
    ax.axhline(overall * 100, color=config.COLOR_TEXT_MUTED, linestyle="--", linewidth=1)
    for s, row in t.nlargest(4, "fraud_rate_pct").iterrows():
        ax.annotate(f" {s} ({int(row.n_cards)} card{'s' if row.n_cards > 1 else ''})",
                    (row.n_cards, row.fraud_rate_pct), fontsize=8, color=config.COLOR_TEXT_MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of distinct cards in state (log scale)")
    ax.set_ylabel("Fraud rate (%, log scale)")
    ax.set_title("State fraud rate vs number of cards: extremes come from tiny states")
    save(fig, FIG / "05_fraud_rate_by_state.png")
    return t.sort_values("fraud_rate_pct", ascending=False)


def plot_age(train: pd.DataFrame, overall: float) -> pd.DataFrame:
    bands = pd.cut(train["age"], bins=[0, 25, 35, 45, 55, 65, 75, 120],
                   labels=["<25", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"], right=False)
    t = _rate_table(train.assign(age_band=bands), "age_band")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(t.index.astype(str), t["fraud_rate_pct"], color=config.COLOR_FRAUD, width=0.6)
    ax.axhline(overall * 100, color=config.COLOR_TEXT_MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Cardholder age at transaction")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud rate by age band (train)")
    ax.grid(axis="x", visible=False)
    save(fig, FIG / "06_fraud_rate_by_age.png")
    return t


def plot_monthly(train: pd.DataFrame) -> pd.DataFrame:
    t = _rate_table(train, "month")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].bar(t.index, t["n_txn"], color=config.COLOR_LEGIT, width=0.7)
    axes[0].set_title("Monthly transaction volume (train)")
    axes[0].set_ylabel("Transactions")
    axes[0].grid(axis="x", visible=False)
    axes[1].plot(t.index, t["fraud_rate_pct"], color=config.COLOR_FRAUD, marker="o", markersize=4)
    axes[1].set_title("Monthly fraud rate (train)")
    axes[1].set_ylabel("Fraud rate (%)")
    axes[1].tick_params(axis="x", rotation=60)
    save(fig, FIG / "07_monthly_trend.png")
    return t


ENGINEERED = ["card_txn_count_24h", "card_txn_count_1h", "amt_to_card_mean_ratio", "amt_zscore_card",
              "secs_since_prev_txn", "card_category_share", "card_prior_txn_count"]


def plot_engineered(train: pd.DataFrame) -> pd.DataFrame:
    """ECDF per class: an ECDF shows the whole distribution without binning choices."""
    show = ["card_txn_count_24h", "amt_to_card_mean_ratio", "secs_since_prev_txn", "card_category_share"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, col in zip(axes, show):
        for label, color, val in [("Legitimate", config.COLOR_LEGIT, 0), ("Fraud", config.COLOR_FRAUD, 1)]:
            x = np.sort(train.loc[train[config.TARGET] == val, col].dropna().to_numpy())
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=color, label=label)
        if col in ("amt_to_card_mean_ratio", "secs_since_prev_txn"):
            ax.set_xscale("log")
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("Cumulative share")
    axes[0].legend()
    save(fig, FIG / "08_engineered_feature_separation.png")

    y = train[config.TARGET].to_numpy()
    rows = []
    for col in ENGINEERED + ["amt", "age", "city_pop", "hour"]:
        s = train[col]
        auc = rank_auc(y, s.fillna(s.median()).to_numpy())
        rows.append({"feature": col, "univariate_auc": round(auc, 4), "separation": round(abs(auc - 0.5) * 2, 4),
                     "median_legit": train.loc[y == 0, col].median(), "median_fraud": train.loc[y == 1, col].median()})
    return pd.DataFrame(rows).sort_values("separation", ascending=False)


def plot_numeric(train: pd.DataFrame) -> None:
    cols = [("age", False), ("city_pop", True), ("hour", False)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (col, logx) in zip(axes, cols):
        s = train[col]
        bins = np.logspace(np.log10(max(s.min(), 1)), np.log10(s.max()), 50) if logx else 40
        if col == "hour":
            bins = np.arange(25) - 0.5
        for label, color, val in [("Legitimate", config.COLOR_LEGIT, 0), ("Fraud", config.COLOR_FRAUD, 1)]:
            ax.hist(s[train[config.TARGET] == val], bins=bins, density=True, histtype="step",
                    color=color, linewidth=2, label=label)
        if logx:
            ax.set_xscale("log")
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("Density (within class)")
    axes[0].legend()
    save(fig, FIG / "09_numeric_distributions.png")


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


def run() -> None:
    apply_style()
    full = build_feature_table()
    train = full[full["split"] == "train"]
    overall = train[config.TARGET].mean()

    plot_class_distribution(full)
    plot_amount(train)
    cat = plot_category(train, overall)
    hour, dow = plot_time(train, overall)
    state = plot_state(train, overall)
    age = plot_age(train, overall)
    monthly = plot_monthly(train)
    eng = plot_engineered(train)
    plot_numeric(train)

    amt_stats = train.groupby(config.TARGET)["amt"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    lines = ["# EDA summary (train split only)", "",
             "_Auto-generated by `python -m src.eda`. All numbers computed from the train split._", "",
             "## Figures and the question each answers", ""]
    lines += [f"- `figures/{k}` — {v}" for k, v in FIGURE_QUESTIONS.items()]
    lines += ["", f"Overall train fraud rate: **{overall:.4%}**", "",
              "## Amount by class ($)", "", _md(amt_stats, ".2f"), "",
              "## Fraud rate by category", "", _md(cat.sort_values("fraud_rate_pct", ascending=False)), "",
              "## Fraud rate by hour", "", _md(hour), "",
              "## Fraud rate by day of week (0=Mon)", "", _md(dow), "",
              "## Fraud rate by age band", "", _md(age), "",
              "## State: fraud rate vs number of cards (top 10 by rate)", "", _md(state.head(10)), "",
              f"States with ≤ 3 cards: **{int((state['n_cards'] <= 3).sum())} of {len(state)}**", "",
              "## Monthly volume and fraud rate", "", _md(monthly), "",
              "## Univariate separation of model inputs (train)", "",
              "AUC of each feature used alone as a score (missing → median). `separation` = |AUC−0.5|×2.", "",
              _md(eng.set_index("feature")), ""]
    (config.REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines))
    logger.info("EDA figures written to %s", FIG)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
