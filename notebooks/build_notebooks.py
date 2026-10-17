"""Build and execute the walkthrough notebooks from the `src/` modules.

Usage (from project root, after the pipeline has been run):
    python notebooks/build_notebooks.py

The notebooks are thin narrative wrappers: all logic lives in `src/` so there is
one tested implementation. Narrative cells describe methods; every number shown
comes from an executed code cell.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

NB_DIR = Path(__file__).resolve().parent
SETUP = """import sys, pathlib
ROOT = pathlib.Path.cwd().resolve().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import Image, Markdown, display
pd.set_option("display.max_columns", 50); pd.set_option("display.width", 200)
from src import config"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


NOTEBOOKS: dict[str, list] = {
    "01_data_understanding.ipynb": [
        md("""# 01 · Data understanding
Goal: know exactly what is in the data **before** any modelling: structure, types, missing values,
duplicates, the target, identifiers, timestamps and leakage risks. Data is loaded with the Hugging Face
`datasets` library (`src/data_loader.py`) and cached locally as Parquet."""),
        code(SETUP + "\nfrom src.data_loader import load_splits\nfrom src import inspect_data as insp\nsplits = load_splits()\n{k: v.shape for k, v in splits.items()}"),
        md("## Columns, types, missing values, cardinality (train)"),
        code("train, test = splits['train'], splits['test']\ninsp.column_profile(train)"),
        md("## Duplicates\nFull-row duplicates, and duplicates after removing the row-index column."),
        code("{n: (int(df.duplicated().sum()), int(df.drop(columns=['Unnamed: 0']).duplicated().sum())) for n, df in splits.items()}"),
        md("## Target and class imbalance"),
        code("pd.DataFrame({n: df[config.TARGET].value_counts() for n, df in splits.items()}).assign(\n    train_pct=lambda d: 100*d['train']/d['train'].sum(), test_pct=lambda d: 100*d['test']/d['test'].sum())"),
        md("## Identifiers and timestamps\nCandidate identifiers are flagged by name pattern or near-unique values; they need human judgement."),
        code("display(insp.candidate_identifiers(train, config.TARGET))\ninsp.detect_datetime_columns(train)"),
        md("## Train/test relationship\nIs the split by time? Do rows or IDs overlap?"),
        code("tr_t, te_t = pd.to_datetime(train[config.TIME_COL]), pd.to_datetime(test[config.TIME_COL])\nprint('train', tr_t.min(), '->', tr_t.max()); print('test ', te_t.min(), '->', te_t.max())\ninsp.cross_split_overlap(splits, ['trans_num', 'cc_num'], config.TARGET)"),
        md("## Card = customer?\nEach card number maps to a single customer profile, so `cc_num` is the natural key for behavioural features."),
        code("prof = ['first','last','street','city','state','zip','dob','job']\nboth = pd.concat([train, test])\nprint('max distinct profile values per card:', both.groupby('cc_num')[prof].nunique().max().max())\nprint('cards:', both.cc_num.nunique())"),
        md("## The fraud-only-card artefact\nCards whose transactions are *all* fraud, and cards that appear only in test."),
        code("g = both.groupby(['cc_num'])[config.TARGET].agg(['mean','size'])\nnew = set(test.cc_num) - set(train.cc_num)\nprint('cards that are 100% fraud:', int((g['mean'] == 1).sum()))\nprint('test-only cards:', len(new), '| their fraud share:', g.loc[list(new), 'mean'].mean())"),
        md("Full write-ups: `reports/data_inspection.md`, `reports/data_inspection_notes.md`, `reports/data_leakage_analysis.md`."),
    ],
    "02_eda.ipynb": [
        md("""# 02 · Exploratory data analysis (train split only)
Each figure answers one question. The test split is excluded so no decision is influenced by it.
Figures are produced by `python -m src.eda`."""),
        code(SETUP + "\nfrom src.eda import FIGURE_QUESTIONS\nfrom src.features import build_feature_table\ndf = build_feature_table(); train = df[df.split == 'train']"),
        code("for fname, question in FIGURE_QUESTIONS.items():\n    display(Markdown(f'### {question}'))\n    display(Image(filename=str(config.FIGURES_DIR / fname)))"),
        md("## Key numbers behind the figures"),
        code("display(train.groupby(config.TARGET)['amt'].describe(percentiles=[.1,.25,.5,.75,.9]).round(2))\ntrain.groupby('category')[config.TARGET].agg(['size','sum','mean']).sort_values('mean', ascending=False).round(4)"),
        code("train.groupby('hour')[config.TARGET].mean().mul(100).round(3).to_frame('fraud_rate_pct').T"),
        md("Summary tables: `reports/eda_summary.md`."),
    ],
    "03_feature_engineering.ipynb": [
        md("""# 03 · Feature engineering and preprocessing
Card-history features use **only earlier transactions of the same card** and **never the fraud label**
(labels arrive weeks later in reality). Implementation: `src/features.py`; preprocessing: `src/preprocessing.py`."""),
        code(SETUP + "\nfrom src.features import build_feature_table, model_features\ndf = build_feature_table()\nmodel_features()"),
        md("## Feature catalogue\n| Feature | Definition | Why it may help |\n|---|---|---|\n| `amt` | transaction amount | fraud amounts are bimodal: small card tests and large purchases |\n| `category` | merchant category | online shopping / misc_net / grocery_pos carry higher fraud rates |\n| `hour`, `day_of_week`, `is_weekend` | from timestamp | fraud concentrates late at night |\n| `age` | age at transaction | fraud rate varies by age band |\n| `city_pop` | cardholder city population | proxy for urban/rural context |\n| `card_txn_count_1h`, `card_txn_count_24h` | earlier txns of the card in the window | fraud comes in bursts |\n| `secs_since_prev_txn` | time since the card's previous txn | bursts mean short gaps |\n| `amt_to_card_mean_ratio` | amount / card's average so far | unusual spend for *this* customer |\n| `amt_zscore_card` | (amount − card mean) / card std | same idea, scale-aware |\n| `card_category_share` | share of earlier txns in this category | unfamiliar category for this customer |\n| `card_prior_txn_count` | number of earlier txns | account tenure (sensitive to a dataset artefact; ablated) |"),
        md("## Verification against a brute-force recomputation for one card"),
        code("c = df.cc_num.iloc[0]; d = df[df.cc_num == c].sort_values('ts').reset_index(drop=True); i = 500; t = d.ts[i]\nprint('24h count  naive:', int(((d.ts >= t - pd.Timedelta('24h')) & (d.index < i)).sum()), '| feature:', d.card_txn_count_24h[i])\nprint('mean ratio naive: %.4f | feature: %.4f' % (d.amt[i] / d.amt[:i].mean(), d.amt_to_card_mean_ratio[i]))"),
        md("## How well does each feature separate the classes on its own? (train)"),
        code("from src.inspect_data import rank_auc\ntr = df[df.split == 'train']; y = tr[config.TARGET].to_numpy()\npd.DataFrame({f: {'univariate_auc': rank_auc(y, tr[f].fillna(tr[f].median()).to_numpy())} for f in model_features() if f != 'category'}).T.sort_values('univariate_auc')"),
        md("## Preprocessing pipelines\nLinear model: impute → signed log for skewed columns → standardise; one-hot for category/hour/weekday.\nTrees: impute only; ordinal-encode category. Both are fitted inside a `Pipeline`, i.e. on training folds only."),
        code("from src.preprocessing import build_preprocessor\nfor kind in ('linear', 'tree'):\n    pre = build_preprocessor(kind, model_features()).fit(tr[model_features()].head(50000))\n    print(kind, '->', len(pre.get_feature_names_out()), 'model inputs'); print(list(pre.get_feature_names_out())[:20], '...')"),
    ],
    "04_model_training.ipynb": [
        md("""# 04 · Model training, comparison, tuning and threshold selection
Training is run with `python -m src.train` (takes a while); this notebook presents its saved outputs.
* CV: 5-fold **StratifiedGroupKFold** grouped by card on a stratified 30% sample of train.
* Tuning: RandomizedSearchCV scored on PR-AUC.
* Threshold and risk tiers: chosen on out-of-fold predictions over the full train set.
* Test set (later time period): used once at the end."""),
        code(SETUP + "\nR = config.REPORTS_DIR"),
        md("## Class-imbalance strategies (metrics at threshold 0.5)"),
        code("pd.read_csv(R / 'imbalance_strategy_comparison.csv')[['model','strategy','precision','recall','f1','roc_auc','pr_auc','pr_auc_std']].round(4)"),
        md("## Model comparison (`reports/model_comparison.csv`)"),
        code("pd.read_csv(R / 'model_comparison.csv').round(4)"),
        code("display(Image(filename=str(config.FIGURES_DIR / 'model_cv_pr_roc.png')))"),
        md("## Hyperparameter search (top 5 XGBoost candidates)"),
        code("t = pd.read_csv(R / 'tuning_xgboost.csv')\nt[[c for c in t.columns if c.startswith('param_')] + ['mean_test_score','std_test_score']].head().round(4)"),
        md("## Threshold analysis (out-of-fold, full train)"),
        code("display(pd.read_csv(R / 'threshold_analysis.csv').round(4))\ndisplay(Image(filename=str(config.FIGURES_DIR / 'threshold_tradeoff.png')))"),
        md("## Final evaluation on the out-of-time test set"),
        code("display(pd.read_csv(R / 'test_metrics.csv').round(4))\ndisplay(Image(filename=str(config.FIGURES_DIR / 'model_test_pr_roc.png')))\ndisplay(Image(filename=str(config.FIGURES_DIR / 'confusion_matrix_xgb_test.png')))"),
        md("## Robustness: does performance rely on the new-card artefact?"),
        code("pd.read_csv(R / 'robustness_checks.csv').round(4)"),
    ],
    "05_model_explainability.ipynb": [
        md("""# 05 · Explainability and risk scoring
SHAP values decompose each prediction (in log-odds) into additive feature contributions. They describe
how the **model** uses features, not what **causes** fraud. Figures come from `python -m src.explainability`."""),
        code(SETUP),
        code("for f in ['xgb_gain_importance.png', 'shap_importance.png', 'shap_summary.png']:\n    display(Image(filename=str(config.SHAP_FIGURES_DIR / f)))"),
        md("## Individual explanations: a caught fraud and a false alert"),
        code("for label in ['true_positive', 'false_positive']:\n    display(Markdown(f'### {label}'))\n    display(Image(filename=str(config.SHAP_FIGURES_DIR / f'waterfall_{label}.png')))\n    display(pd.read_csv(config.REPORTS_DIR / f'shap_example_{label}.csv').head(8))"),
        md("## Risk tiers on the test period"),
        code("from src.risk import load_thresholds\nprint(load_thresholds())\npd.read_csv(config.REPORTS_DIR / 'risk_tier_summary.csv').round(3)"),
    ],
}


def main() -> None:
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
        ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
        nbf.write(nb, NB_DIR / name)
        print("built", name)


if __name__ == "__main__":
    main()
