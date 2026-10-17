"""Evaluation utilities: metrics, threshold analysis and evaluation plots."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config
from src.plotting import save


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    tn: int
    fp: int
    fn: int
    tp: int
    threshold: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> Metrics:
    """Threshold-free metrics (ROC-AUC, PR-AUC) plus metrics at `threshold`.

    PR-AUC is computed as average precision (step-wise area under the PR curve),
    which avoids the optimistic linear interpolation of the trapezoid rule.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return Metrics(
        accuracy=accuracy_score(y_true, pred),
        precision=precision_score(y_true, pred, zero_division=0),
        recall=recall_score(y_true, pred, zero_division=0),
        f1=f1_score(y_true, pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, proba),
        pr_auc=average_precision_score(y_true, proba),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        threshold=float(threshold),
    )


# ------------------------------------------------------------------ thresholds
def threshold_table(y_true, proba, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows = [compute_metrics(y_true, proba, t).as_dict() for t in thresholds]
    return pd.DataFrame(rows)[["threshold", "precision", "recall", "f1", "fp", "fn", "tp", "tn"]]


def threshold_for_min_precision(y_true, proba, min_precision: float) -> float:
    """Lowest threshold (=> highest recall) whose precision is >= min_precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(precision[:-1] >= min_precision)[0]
    if len(ok) == 0:
        raise ValueError(f"No threshold reaches precision {min_precision}")
    best = ok[np.argmax(recall[:-1][ok])]
    return float(thresholds[best])


def threshold_for_recall(y_true, proba, target_recall: float) -> float:
    """Highest threshold whose recall is still >= target_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(recall[:-1] >= target_recall)[0]
    return float(thresholds[ok.max()])


def threshold_max_f1(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


# ------------------------------------------------------------------ plots
MODEL_COLORS = {"Logistic Regression": "#2a78d6", "Random Forest": "#1baf7a", "XGBoost": "#eb6834"}


def plot_pr_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], title_suffix: str, path) -> None:
    """curves: {model_name: (y_true, proba)}."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    base_rate = None
    for name, (y, p) in curves.items():
        color = MODEL_COLORS.get(name, config.COLOR_TEXT_MUTED)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(rec, prec, color=color, label=f"{name} (PR-AUC {average_precision_score(y, p):.3f})")
        fpr, tpr, _ = roc_curve(y, p)
        axes[1].plot(fpr, tpr, color=color, label=f"{name} (ROC-AUC {roc_auc_score(y, p):.3f})")
        base_rate = np.mean(y)
    axes[0].axhline(base_rate, color=config.COLOR_TEXT_MUTED, linestyle="--", linewidth=1)
    axes[0].annotate(f"random classifier = fraud rate ({base_rate:.2%})", (0.02, base_rate + 0.02),
                     fontsize=8, color=config.COLOR_TEXT_MUTED)
    axes[0].set(xlabel="Recall", ylabel="Precision", title=f"Precision–recall {title_suffix}", ylim=(0, 1.02))
    axes[1].plot([0, 1], [0, 1], color=config.COLOR_TEXT_MUTED, linestyle="--", linewidth=1)
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC {title_suffix}")
    for ax in axes:
        ax.legend(loc="lower left" if ax is axes[0] else "lower right", fontsize=8)
    save(fig, path)


def plot_confusion(m: Metrics, title: str, path) -> None:
    cm = np.array([[m.tn, m.fp], [m.fn, m.tp]])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    shares = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({shares[i, j]:.1%} of row)", ha="center", va="center",
                    color="white" if shares[i, j] > 0.6 else config.COLOR_TEXT, fontsize=9)
    ax.set_xticks([0, 1], ["Pred legit", "Pred fraud"])
    ax.set_yticks([0, 1], ["Actual legit", "Actual fraud"])
    ax.grid(False)
    ax.set_title(title, fontsize=10)
    save(fig, path)


def plot_threshold_tradeoff(table: pd.DataFrame, chosen: float, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for col, color in [("precision", "#2a78d6"), ("recall", "#eb6834"), ("f1", "#1baf7a")]:
        axes[0].plot(table["threshold"], table[col], color=color, label=col.title())
    axes[0].set(xlabel="Probability threshold", ylabel="Score", title="Metrics vs threshold (validation, out-of-fold)")
    axes[1].plot(table["threshold"], table["fp"], color="#2a78d6", label="False positives (false alerts)")
    axes[1].plot(table["threshold"], table["fn"], color="#eb6834", label="False negatives (missed fraud)")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Probability threshold", ylabel="Count (log scale)", title="Errors vs threshold")
    for ax in axes:
        ax.axvline(chosen, color=config.COLOR_TEXT_MUTED, linestyle="--", linewidth=1)
        ax.legend(fontsize=8)
    axes[0].annotate(f" chosen = {chosen:.3f}", (chosen, 0.05), fontsize=8, color=config.COLOR_TEXT_MUTED)
    save(fig, path)
