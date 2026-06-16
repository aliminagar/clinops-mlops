"""End-to-end test for the training orchestrator on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from clinops.config import settings
from clinops.training import train

TARGET = settings.target_name


def _write_store(feature_store: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "features.parquet"
    feature_store.to_parquet(path)
    return path


def test_run_training_end_to_end(feature_store: pd.DataFrame, tmp_path: Path) -> None:
    feature_path = _write_store(feature_store, tmp_path)
    summary = train.run_training(
        feature_path=feature_path,
        splits_path=tmp_path / "splits.parquet",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        seed=settings.random_seed,
        k=3,
        run_shap=False,
        track=False,
    )

    # Persisted artifacts exist.
    assert (tmp_path / "models" / "logistic_regression.joblib").exists()
    assert (tmp_path / "models" / "lightgbm.joblib").exists()
    metrics_file = tmp_path / "reports" / "metrics.json"
    assert metrics_file.exists()
    assert (tmp_path / "reports" / "calibration_test.png").exists()

    # Metrics JSON parses and has the expected structure.
    on_disk = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert on_disk == summary
    assert summary["headline_metric"] == "pr_auc"
    assert summary["dataset"]["dropped_zero_variance"] == ["cond_copd"]
    assert sum(summary["dataset"]["split_counts"].values()) == len(feature_store)

    # Three candidates trained on the same protocol, including the torch challenger.
    assert set(summary["models"]) == {"logistic_regression", "lightgbm", "torch_mlp"}
    for model in summary["models"].values():
        assert model["cv_train"]["k"] == 3
        assert 0.0 <= model["cv_train"]["mean"] <= 1.0
        assert 0.0 <= model["val"]["pr_auc"] <= 1.0
        assert 0.0 <= model["test"]["pr_auc"] <= 1.0
    # LightGBM carries a train-derived scale_pos_weight; logreg does not.
    assert summary["models"]["lightgbm"]["scale_pos_weight"] > 1.0
    assert summary["models"]["logistic_regression"]["scale_pos_weight"] is None
    assert summary["mlflow"] is None  # tracking disabled in this test

    # The champion is the highest-CV-PR-AUC model among the three.
    assert summary["champion"]["model"] in summary["models"]
    best = max(summary["models"], key=lambda n: summary["models"][n]["cv_train"]["mean"])
    assert summary["champion"]["model"] == best

    # All three operating points are reported side by side; @0.5 is retained.
    for name in ("logistic_regression", "lightgbm", "torch_mlp"):
        block = summary["models"][name]
        assert set(block["operating_points"]) == {"fixed_0p5", "max_f1", "f2"}
        for point in block["operating_points"].values():
            assert {"threshold", "precision", "recall", "f1"} <= set(point)
        assert block["operating_points"]["f2"]["deployed"] is True
        assert block["test"]["threshold"] == 0.5  # fixed @0.5 not replaced
        # The deployed point is F2 (recall-weighted) by default.
        assert block["deployed_threshold"]["rule"] == "max_fbeta"
        assert block["deployed_threshold"]["beta"] == 2.0


def test_run_training_is_deterministic(feature_store: pd.DataFrame, tmp_path: Path) -> None:
    feature_path = _write_store(feature_store, tmp_path)
    common: dict[str, Any] = {
        "feature_path": feature_path,
        "splits_path": tmp_path / "splits.parquet",
        "seed": settings.random_seed,
        "k": 3,
        "run_shap": False,
        "track": False,
    }
    first = train.run_training(models_dir=tmp_path / "m1", reports_dir=tmp_path / "r1", **common)
    second = train.run_training(models_dir=tmp_path / "m2", reports_dir=tmp_path / "r2", **common)

    for name in ("logistic_regression", "lightgbm"):
        a, b = first["models"][name], second["models"][name]
        assert a["test"]["pr_auc"] == b["test"]["pr_auc"]
        assert a["cv_train"]["mean"] == b["cv_train"]["mean"]
