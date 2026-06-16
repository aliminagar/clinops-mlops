"""Tests for the PyTorch challenger (sklearn-compatible wrapper)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinops.config import settings
from clinops.models.torch_model import TorchMLPClassifier
from clinops.training import evaluate

TARGET = settings.target_name


def _xy(feature_store: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return feature_store.drop(columns=[TARGET, "cond_copd"]), feature_store[TARGET]


def test_fit_predict_proba_in_unit_interval(feature_store: pd.DataFrame) -> None:
    x, y = _xy(feature_store)
    model = TorchMLPClassifier(seed=settings.random_seed, epochs=60).fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_deterministic_across_two_fits(feature_store: pd.DataFrame) -> None:
    x, y = _xy(feature_store)
    p1 = TorchMLPClassifier(seed=7, epochs=50).fit(x, y).predict_proba(x)
    p2 = TorchMLPClassifier(seed=7, epochs=50).fit(x, y).predict_proba(x)
    np.testing.assert_array_equal(p1, p2)


def test_pos_weight_matches_class_ratio(feature_store: pd.DataFrame) -> None:
    x, y = _xy(feature_store)
    model = TorchMLPClassifier(seed=settings.random_seed, epochs=10).fit(x, y)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    assert model.pos_weight_ == (n_neg / n_pos)
    assert model.pos_weight_ > 1.0  # rare positives -> upweighted


def test_integrates_with_cv_and_eval_path(feature_store: pd.DataFrame) -> None:
    x, y = _xy(feature_store)

    def make(_fold_target: pd.Series) -> TorchMLPClassifier:
        return TorchMLPClassifier(seed=settings.random_seed, epochs=40)

    cv = evaluate.cross_val_average_precision(make, x, y, k=3, seed=settings.random_seed)
    assert cv["k"] == 3
    assert len(cv["folds"]) == 3
    assert 0.0 <= cv["mean"] <= 1.0

    model = TorchMLPClassifier(seed=settings.random_seed, epochs=60).fit(x, y)
    metrics = evaluate.compute_metrics(y, model.predict_proba(x)[:, 1])
    assert 0.0 <= metrics["pr_auc"] <= 1.0
