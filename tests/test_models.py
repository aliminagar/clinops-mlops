"""Tests for the baseline model factories (logistic regression + LightGBM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from clinops.config import settings
from clinops.models import sklearn_models

TARGET = settings.target_name


def _xy(feature_store: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return feature_store.drop(columns=[TARGET]), feature_store[TARGET]


def test_logistic_regression_pipeline_fits_and_predicts(feature_store: pd.DataFrame) -> None:
    x, y = _xy(feature_store)
    continuous = ["age", "bmi", "systolic_bp"]
    model = sklearn_models.build_logistic_regression(continuous, seed=settings.random_seed)

    assert isinstance(model, Pipeline)
    assert model.named_steps["classifier"].class_weight == "balanced"

    model.fit(x, y)
    proba = model.predict_proba(x)[:, 1]
    assert proba.shape == (len(x),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_lightgbm_uses_scale_pos_weight_and_predicts(feature_store: pd.DataFrame) -> None:
    x, y = _xy(feature_store)
    spw = sklearn_models.scale_pos_weight_from(y)
    model = sklearn_models.build_lightgbm(spw, seed=settings.random_seed)

    assert isinstance(model, LGBMClassifier)
    assert model.scale_pos_weight == spw

    model.fit(x, y)
    proba = model.predict_proba(x)[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))


def test_scale_pos_weight_from() -> None:
    assert sklearn_models.scale_pos_weight_from([0, 0, 0, 1]) == 3.0  # 3 neg / 1 pos
    assert sklearn_models.scale_pos_weight_from([0, 0, 1, 1]) == 1.0  # balanced
    # No positives -> safe fallback, never divides by zero.
    assert sklearn_models.scale_pos_weight_from([0, 0, 0]) == 1.0


def test_lightgbm_is_deterministic(feature_store: pd.DataFrame) -> None:
    x, y = _xy(feature_store)
    spw = sklearn_models.scale_pos_weight_from(y)
    p1 = sklearn_models.build_lightgbm(spw, seed=1).fit(x, y).predict_proba(x)[:, 1]
    p2 = sklearn_models.build_lightgbm(spw, seed=1).fit(x, y).predict_proba(x)[:, 1]
    np.testing.assert_allclose(p1, p2)
