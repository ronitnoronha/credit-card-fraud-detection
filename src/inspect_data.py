"""Stage 0 — dataset inspection (no modelling).

Makes no assumptions about column names. Heuristics only *flag* candidate
targets / identifiers / timestamps / leakage; a human confirms them.

Usage:
    python -m src.inspect_data
Writes reports/data_inspection.md.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR
from src.data_loader import load_splits

logger = logging.getLogger(__name__)

TARGET_NAME_PATTERN = re.compile(r"fraud|label|target|class", re.I)
ID_NAME_PATTERN = re.compile(r"(^|_)(id|num|no|number|cc|card|account|acct|trans)($|_)|^unnamed", re.I)
TIME_NAME_PATTERN = re.compile(r"date|time|ts|timestamp|unix|dob", re.I)


def _md_table(df: pd.DataFrame) -> str:
    return df.to_markdown() if not df.empty else "_none_"


def find_target(df: pd.DataFrame) -> str | None:
    """Binary column whose name looks like a label."""
    for col in df.columns:
        if TARGET_NAME_PATTERN.search(col) and df[col].dropna().nunique() == 2:
            return col
    return None


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    prof = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(3),
        "n_unique": df.nunique(dropna=True),
    })
    prof["unique_ratio"] = (prof["n_unique"] / n).round(4)
    prof["example"] = [repr(df[c].dropna().iloc[0])[:40] if df[c].notna().any() else "" for c in df.columns]
    return prof


def detect_datetime_columns(df: pd.DataFrame, sample: int = 2000) -> dict[str, str]:
    """Return {col: reason} for columns that parse as datetimes or look like epoch seconds."""
    found: dict[str, str] = {}
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            found[col] = "datetime dtype"
        elif s.dtype == object or pd.api.types.is_string_dtype(s):
            parsed = pd.to_datetime(s.sample(min(sample, len(s)), random_state=0), errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.95:
                found[col] = f"string parses as datetime (range sample {parsed.min()} → {parsed.max()})"
        elif pd.api.types.is_integer_dtype(s) and TIME_NAME_PATTERN.search(col):
            lo, hi = s.min(), s.max()
            if 9e8 < lo < 2.5e9 and 9e8 < hi < 2.5e9:
                found[col] = f"integer in epoch-seconds range ({pd.to_datetime(lo, unit='s')} → {pd.to_datetime(hi, unit='s')})"
    return found


def candidate_identifiers(df: pd.DataFrame, target: str | None) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col == target:
            continue
        ratio = df[col].nunique(dropna=True) / len(df)
        name_hit = bool(ID_NAME_PATTERN.search(col))
        if ratio > 0.9 or name_hit:
            rows.append({"column": col, "unique_ratio": round(ratio, 4), "name_matches_id_pattern": name_hit})
    return pd.DataFrame(rows)


def rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney rank formulation (no sklearn needed)."""
    ranks = pd.Series(score).rank(method="average").to_numpy()
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def univariate_leakage_scan(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Flag single columns that almost perfectly separate the classes.

    Numeric: |AUC - 0.5| of the raw column. Categorical: AUC of the in-sample
    per-category fraud rate (optimistic for high-cardinality columns, which is
    why unique_ratio is reported alongside).
    """
    y = df[target].astype(int).to_numpy()
    rows = []
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            score = s.fillna(s.median()).to_numpy()
            kind = "numeric"
        else:
            score = s.astype(str).map(df.groupby(s.astype(str))[target].mean()).to_numpy()
            kind = "categorical (target-mean)"
        auc = rank_auc(y, score)
        rows.append({"column": col, "kind": kind, "auc": round(auc, 4),
                     "separation": round(abs(auc - 0.5) * 2, 4),
                     "unique_ratio": round(s.nunique() / len(s), 4)})
    return pd.DataFrame(rows).sort_values("separation", ascending=False).reset_index(drop=True)


def cross_split_overlap(splits: dict[str, pd.DataFrame], id_cols: list[str], target: str | None) -> pd.DataFrame:
    """Overlap of identifier values and of exact rows between train and test."""
    if "train" not in splits or "test" not in splits:
        return pd.DataFrame()
    tr, te = splits["train"], splits["test"]
    rows = []
    for col in id_cols:
        if col in tr and col in te:
            shared = set(tr[col].dropna().unique()) & set(te[col].dropna().unique())
            rows.append({"what": f"values of `{col}`", "n_shared": len(shared),
                         "pct_of_test_unique": round(100 * len(shared) / max(te[col].nunique(), 1), 2)})
    # Exact-row overlap, ignoring pure row-index columns.
    index_like = [c for c in tr.columns if c.lower().startswith("unnamed")]
    common = [c for c in tr.columns if c in te.columns and c not in index_like]
    tr_h = set(pd.util.hash_pandas_object(tr[common], index=False))
    te_h = pd.util.hash_pandas_object(te[common], index=False)
    rows.append({"what": "exact rows (excluding index-like cols)", "n_shared": int(te_h.isin(tr_h).sum()),
                 "pct_of_test_unique": round(100 * te_h.isin(tr_h).mean(), 4)})
    return pd.DataFrame(rows)


def inspect(splits: dict[str, pd.DataFrame]) -> str:
    out: list[str] = ["# Dataset inspection report", "",
                      "_Auto-generated by `python -m src.inspect_data`. Heuristic flags need human review._", ""]

    out += ["## 1–2. Structure and split sizes", ""]
    out.append(_md_table(pd.DataFrame({k: {"rows": v.shape[0], "cols": v.shape[1]} for k, v in splits.items()}).T))
    names = list(splits)
    cols_equal = all(list(splits[n].columns) == list(splits[names[0]].columns) for n in names)
    out += ["", f"Identical column lists across splits: **{cols_equal}**", ""]

    ref_name = "train" if "train" in splits else names[0]
    ref = splits[ref_name]
    target = find_target(ref)

    out += [f"## 3–5, 9. Columns, dtypes, missing values, cardinality (`{ref_name}`)", "", _md_table(column_profile(ref)), ""]
    for n in names:
        if n != ref_name:
            miss = splits[n].isna().sum()
            out += [f"Missing values in `{n}`: total **{int(miss.sum())}**"
                    + (f" — {miss[miss > 0].to_dict()}" if miss.sum() else ""), ""]

    out += ["## 6. Duplicate rows", ""]
    for n, df in splits.items():
        index_like = [c for c in df.columns if c.lower().startswith("unnamed")]
        content = df.drop(columns=index_like)
        out.append(f"- `{n}`: full-row duplicates = **{int(df.duplicated().sum())}**; "
                   f"duplicates ignoring index-like cols {index_like} = **{int(content.duplicated().sum())}**")
    out.append("")

    out += ["## 7–8. Target column and class distribution", ""]
    if target is None:
        out += ["No binary label-like column found automatically — **needs manual identification**.", ""]
    else:
        out += [f"Candidate target: **`{target}`**", ""]
        dist = pd.DataFrame({n: df[target].value_counts() for n, df in splits.items() if target in df})
        pct = pd.DataFrame({f"{n} %": (df[target].value_counts(normalize=True) * 100).round(4)
                            for n, df in splits.items() if target in df})
        out += [_md_table(pd.concat([dist, pct], axis=1)), ""]
        for n, df in splits.items():
            if target in df:
                vc = df[target].value_counts()
                out.append(f"- `{n}` imbalance ratio (majority:minority) ≈ **{vc.max() / vc.min():.1f} : 1**")
        out.append("")

    out += ["## 9b. Categorical (non-numeric) columns", ""]
    cat_cols = [c for c in ref.columns if not pd.api.types.is_numeric_dtype(ref[c])]
    num_cols = [c for c in ref.columns if pd.api.types.is_numeric_dtype(ref[c])]
    out += [f"Numeric dtype columns ({len(num_cols)}): {num_cols}", "",
            f"Non-numeric dtype columns ({len(cat_cols)}): {cat_cols}", ""]
    for c in cat_cols:
        top = ref[c].value_counts().head(5)
        out.append(f"- `{c}`: {ref[c].nunique()} unique; top 5 = {top.to_dict()}")
    out.append("")

    out += ["## 10, 13. Potential identifiers (customer / card / transaction)", ""]
    ids = candidate_identifiers(ref, target)
    out += [_md_table(ids), ""]

    out += ["## 12. Timestamp columns", ""]
    dt = detect_datetime_columns(ref)
    out += ([f"- `{c}`: {r}" for c, r in dt.items()] or ["_none detected_"]) + [""]
    if dt and "test" in splits and "train" in splits:
        for c, r in dt.items():
            if "dob" in c.lower() or not (ref[c].dtype == object or pd.api.types.is_string_dtype(ref[c])):
                continue
            tr_t = pd.to_datetime(splits["train"][c], errors="coerce", format="mixed")
            te_t = pd.to_datetime(splits["test"][c], errors="coerce", format="mixed")
            out.append(f"- `{c}` train range {tr_t.min()} → {tr_t.max()}; test range {te_t.min()} → {te_t.max()}; "
                       f"test strictly after train: **{te_t.min() > tr_t.max()}**")
        out.append("")

    out += ["## 11. Leakage checks", "", "### Train/test overlap", ""]
    out += [_md_table(cross_split_overlap(splits, ids["column"].tolist() if not ids.empty else [], target)), ""]
    if target is not None:
        out += ["### Univariate separation scan (train)", "",
                "AUC of each column used alone as a score. `separation` = |AUC−0.5|×2 "
                "(0 = useless, 1 = perfect). Values near 1 are leakage suspects; for "
                "high-`unique_ratio` categoricals the target-mean score is in-sample and overfit by construction.", "",
                _md_table(univariate_leakage_scan(ref, target)), ""]
    return "\n".join(out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    splits = load_splits()
    report = inspect(splits)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "data_inspection.md"
    path.write_text(report)
    logger.info("Wrote %s", path)
    print(report)


if __name__ == "__main__":
    main()
