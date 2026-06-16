"""Tests for the imbalance-aware evaluation harness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clinops.config import settings
from clinops.models import sklearn_models
from clinops.training import evaluate

TARGET = settings.target_name


def test_compute_metrics_keys_and_perfect_scores() -> None:
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    metrics = evaluate.compute_metrics(y_true, y_score, threshold=0.5)

    assert "accuracy" not in metrics  # accuracy intentionally not reported
    assert evaluate.HEADLINE_METRIC == "pr_auc"
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["n"] == 4.0
    assert metrics["n_positive"] == 2.0
    assert metrics["prevalence"] == 0.5


def test_compute_metrics_threshold_affects_recall() -> None:
    y_true = [0, 1, 1, 1]
    y_score = [0.2, 0.4, 0.6, 0.9]
    high = evaluate.compute_metrics(y_true, y_score, threshold=0.5)
    assert high["recall"] == pytest.approx(2 / 3)  # only 0.6, 0.9 cross 0.5
    low = evaluate.compute_metrics(y_true, y_score, threshold=0.3)
    assert low["recall"] == 1.0  # all positives cross 0.3


def test_cross_val_average_precision(feature_store: pd.DataFrame) -> None:
    x = feature_store.drop(columns=[TARGET, "cond_copd"])
    y = feature_store[TARGET]

    def make(_fold_target: pd.Series) -> object:
        return sklearn_models.build_logistic_regression(
            ["age", "bmi", "systolic_bp"], seed=settings.random_seed
        )

    cv = evaluate.cross_val_average_precision(make, x, y, k=5, seed=settings.random_seed)
    assert cv["k"] == 5
    assert len(cv["folds"]) == 5
    assert cv["std"] >= 0.0
    prevalence = float(y.mean())
    # A model with signal should beat the no-skill PR-AUC baseline (the prevalence).
    assert prevalence < cv["mean"] <= 1.0


def test_select_threshold_max_fbeta_separable() -> None:
    # Perfectly separable scores: the chosen threshold should split the classes,
    # yielding precision = recall = F1 = 1 on validation.
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    chosen = evaluate.select_threshold(y_true, y_score, rule="max_fbeta", beta=1.0)
    assert chosen["rule"] == "max_fbeta"
    assert chosen["source"] == "validation"
    assert chosen["beta"] == 1.0
    assert 0.3 < chosen["threshold"] <= 0.7
    assert chosen["precision"] == 1.0
    assert chosen["recall"] == 1.0
    assert chosen["fbeta"] == 1.0


def test_select_threshold_beta_trades_recall_for_precision() -> None:
    # With overlap, a large beta (recall-weighted) should not pick a stricter
    # threshold than a small beta (precision-weighted).
    y_true = [0, 0, 1, 0, 1, 1]
    y_score = [0.2, 0.4, 0.45, 0.5, 0.55, 0.8]
    high_recall = evaluate.select_threshold(y_true, y_score, rule="max_fbeta", beta=2.0)
    high_precision = evaluate.select_threshold(y_true, y_score, rule="max_fbeta", beta=0.5)
    assert high_recall["threshold"] <= high_precision["threshold"]
    assert high_recall["recall"] >= high_precision["recall"]


def test_select_threshold_min_precision_floor() -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.6, 0.55, 0.8, 0.9]
    chosen = evaluate.select_threshold(y_true, y_score, rule="min_precision", precision_floor=0.8)
    assert chosen["rule"] == "min_precision"
    assert chosen["precision_floor"] == 0.8
    assert chosen["met_precision_floor"] is True
    assert chosen["precision"] >= 0.8


def test_select_threshold_precision_floor_unreachable_falls_back() -> None:
    # The lone positive is the lowest-scored sample, so max achievable precision
    # is 0.25 — an impossible 0.99 floor triggers the documented fallback.
    y_true = [1, 0, 0, 0]
    y_score = [0.1, 0.4, 0.6, 0.9]
    chosen = evaluate.select_threshold(y_true, y_score, rule="min_precision", precision_floor=0.99)
    assert chosen["met_precision_floor"] is False
    assert "threshold" in chosen


def test_default_operating_point_is_f2() -> None:
    # The deployed default is recall-weighted F2 (max F-beta with beta=2).
    assert settings.threshold_beta == 2.0
    assert evaluate.RULE_MAX_FBETA == "max_fbeta"


def test_f2_selects_higher_recall_than_f1() -> None:
    # A hard positive sits below several negatives: F1 stops early (precision),
    # F2 lowers the threshold to capture it (recall) — the whole point of F2.
    y_true = [1, 1, 0, 0, 1, 0, 0, 0]
    y_score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.3]
    f1 = evaluate.select_threshold(y_true, y_score, rule="max_fbeta", beta=1.0)
    f2 = evaluate.select_threshold(y_true, y_score, rule="max_fbeta", beta=2.0)
    assert f2["recall"] > f1["recall"]
    assert f2["threshold"] < f1["threshold"]


def test_select_and_apply_threshold_uses_val_threshold_on_test() -> None:
    y_val = [0, 0, 1, 1]
    val_score = [0.1, 0.2, 0.8, 0.9]
    y_test = [0, 1, 1]
    test_score = [0.3, 0.85, 0.95]
    record = evaluate.select_and_apply_threshold(
        y_val, val_score, y_test, test_score, rule="max_fbeta", beta=2.0
    )
    assert record["source"] == "validation"
    assert record["beta"] == 2.0
    # The val-selected threshold (~0.5) applied to test recovers both positives.
    assert record["recall"] == 1.0
    assert set(record) >= {"threshold", "precision", "recall", "f1", "val_recall"}


def test_operating_point_matches_threshold() -> None:
    y_true = [0, 1, 1, 1]
    y_score = [0.2, 0.4, 0.6, 0.9]
    point = evaluate.operating_point(y_true, y_score, threshold=0.5)
    assert point["threshold"] == 0.5
    assert point["recall"] == pytest.approx(2 / 3)


def test_plot_calibration_writes_file(tmp_path: Path) -> None:
    curves = {"m": (np.array([0.1, 0.5, 0.9]), np.array([0.15, 0.45, 0.85]))}
    out = evaluate.plot_calibration(curves, tmp_path / "cal.png")
    assert out.exists() and out.stat().st_size > 0


def test_shap_summary_writes_file(feature_store: pd.DataFrame, tmp_path: Path) -> None:
    pytest.importorskip("shap")
    x = feature_store.drop(columns=[TARGET, "cond_copd"])
    y = feature_store[TARGET]
    model = sklearn_models.build_lightgbm(
        sklearn_models.scale_pos_weight_from(y), seed=settings.random_seed
    )
    model.fit(x, y)

    out = evaluate.save_shap_summary(model, x, tmp_path / "shap.png", max_samples=100)
    assert out.exists() and out.stat().st_size > 0
