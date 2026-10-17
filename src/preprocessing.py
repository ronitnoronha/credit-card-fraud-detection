"""Preprocessing pipelines (scikit-learn ColumnTransformer).

Two variants, because the model families need different things:

* linear (Logistic Regression)
    - numeric: median imputation (+ missing-indicator), signed log1p for heavily
      skewed columns, then standardisation. LR is sensitive to feature scale and to
      extreme values, and its coefficients are only comparable on a common scale.
    - category, hour, day_of_week: one-hot. Fraud risk is not monotonic in the hour
      (it spikes 22:00–03:00), which a single linear coefficient cannot represent.

* tree (Random Forest, XGBoost)
    - numeric: median imputation only. Trees split on thresholds, so scaling and
      monotonic transforms change nothing.
    - category: ordinal codes. Trees can isolate any category with a few splits, and
      a single column keeps SHAP explanations readable ("category" instead of 14
      one-hot columns).

Everything is wrapped in a Pipeline together with the model, so `fit` learns
medians / scalers / encoders from training data only (training folds during CV).
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src import config


def signed_log1p(x):
    """log1p that also handles negative values (e.g. z-scores): sign(x)·log(1+|x|)."""
    return np.sign(x) * np.log1p(np.abs(x))


def _split_numeric(features: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    numeric = [f for f in config.NUMERIC_FEATURES if f in features]
    skewed = [f for f in numeric if f in config.SKEWED_FEATURES + ["amt_zscore_card"]]
    plain = [f for f in numeric if f not in skewed]
    cyclic = [f for f in config.CYCLIC_CATEGORICAL_FEATURES if f in features]
    cats = [f for f in config.CATEGORICAL_FEATURES if f in features]
    return skewed, plain, cyclic, cats


def build_preprocessor(kind: str, features: list[str]) -> ColumnTransformer:
    skewed, plain, cyclic, cats = _split_numeric(features)

    if kind == "linear":
        transformers = [
            ("skewed", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]), skewed),
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), plain),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cyclic + cats),
        ]
    elif kind == "tree":
        transformers = [
            ("numeric", SimpleImputer(strategy="median"), skewed + plain),
            ("cyclic", "passthrough", cyclic),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
        ]
    else:
        raise ValueError(f"Unknown preprocessor kind: {kind!r}")

    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")
