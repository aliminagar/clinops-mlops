"""Test the MLflow tracking/registry path on a temporary tracking store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from clinops.config import settings
from clinops.training import train

TARGET = settings.target_name


def test_run_training_logs_to_mlflow(feature_store: pd.DataFrame, tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")

    feature_path = tmp_path / "features.parquet"
    feature_store.to_parquet(feature_path)
    tracking_uri = (tmp_path / "mlruns").as_uri()
    model_name = "clinops-test-model"

    summary = train.run_training(
        feature_path=feature_path,
        splits_path=tmp_path / "splits.parquet",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        seed=settings.random_seed,
        k=3,
        run_shap=False,
        track=True,
        tracking_uri=tracking_uri,
        experiment="clinops-test",
        registered_model_name=model_name,
    )

    refs = summary["mlflow"]
    assert refs is not None
    assert refs["experiment"] == "clinops-test"
    assert refs["parent_run_id"]
    # All three candidates get a nested run.
    assert set(refs["runs"]) == {"logistic_regression", "lightgbm", "torch_mlp"}

    # The champion (highest CV PR-AUC) is registered under the champion alias —
    # any flavor can win.
    registered = refs["registered_model"]
    assert registered["name"] == model_name
    assert int(registered["version"]) >= 1
    assert registered["alias"] == "champion"
    assert registered["champion_model"] in refs["runs"]
    assert registered["champion_model"] == summary["champion"]["model"]

    # The runs and the registered champion version actually exist in the store.
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    parent = client.get_run(refs["parent_run_id"])
    assert parent.info.run_id == refs["parent_run_id"]

    champion_mv = client.get_model_version_by_alias(model_name, "champion")
    assert str(champion_mv.version) == registered["version"]

    # Spot-check the LightGBM child run's logged metrics/params/tags.
    lgbm_run = client.get_run(refs["runs"]["lightgbm"])
    assert lgbm_run.data.metrics["test_pr_auc"] == pytest.approx(
        summary["models"]["lightgbm"]["test"]["pr_auc"]
    )
    assert lgbm_run.data.metrics["operating_threshold"] == pytest.approx(
        summary["models"]["lightgbm"]["deployed_threshold"]["value"]
    )
    assert lgbm_run.data.params["model_type"] == "lightgbm"
    assert lgbm_run.data.tags["operating_point"] == "f2"
    assert lgbm_run.data.params["operating_beta"] == "2.0"
    for point in ("fixed_0p5", "max_f1", "f2"):
        assert f"test_{point}_recall" in lgbm_run.data.metrics
