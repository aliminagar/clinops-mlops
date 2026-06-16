"""Tests for champion/challenger promotion (clinops.registry.promote)."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from clinops.registry import promote

MODEL_NAME = "clinops-promote-test"
# (run name, cv_pr_auc_mean, operating_threshold) — lightgbm has the top CV score.
CANDIDATES = [
    ("logistic_regression", 0.20, 0.50),
    ("lightgbm", 0.40, 0.06),
    ("torch_mlp", 0.31, 0.02),
]


def test_select_champion_picks_highest_cv() -> None:
    assert promote.select_champion({"a": 0.2, "b": 0.5, "c": 0.31}) == "b"
    # Ties resolve deterministically by name.
    assert promote.select_champion({"b": 0.5, "a": 0.5}) == "a"


def _seed_candidate_runs(tmp_path: Path) -> tuple[str, str]:
    """Log a parent run with three candidate child runs to a temp registry."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("promote-test")

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.random((40, 3)), columns=["f0", "f1", "f2"])
    y = (x["f0"] > 0.5).astype(int)
    estimator = LogisticRegression(max_iter=200).fit(x, y)

    with mlflow.start_run() as parent:
        parent_run_id = parent.info.run_id
        for name, cv, threshold in CANDIDATES:
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_metric(promote.CV_METRIC, cv)
                mlflow.log_metric(promote.THRESHOLD_METRIC, threshold)
                mlflow.sklearn.log_model(estimator, name=promote.MODEL_ARTIFACT_PATH)
    return parent_run_id, uri


def test_promote_champion_selects_registers_and_aliases(tmp_path: Path) -> None:
    parent_run_id, uri = _seed_candidate_runs(tmp_path)

    result = promote.promote_champion(parent_run_id, MODEL_NAME, tracking_uri=uri)

    assert result.winner == "lightgbm"  # highest CV PR-AUC (0.40)
    assert result.threshold == pytest.approx(0.06)  # the winner's logged F2 threshold
    assert int(result.version) >= 1

    client = MlflowClient(tracking_uri=uri)
    champion = client.get_model_version_by_alias(MODEL_NAME, promote.CHAMPION_ALIAS)
    assert str(champion.version) == result.version


def test_promote_champion_is_idempotent(tmp_path: Path) -> None:
    parent_run_id, uri = _seed_candidate_runs(tmp_path)

    first = promote.promote_champion(parent_run_id, MODEL_NAME, tracking_uri=uri)
    second = promote.promote_champion(parent_run_id, MODEL_NAME, tracking_uri=uri)

    # Same runs -> same champion and threshold (a re-run just registers a new
    # version of the same champion model, with the alias moved to it).
    assert first.winner == second.winner
    assert first.threshold == second.threshold
    client = MlflowClient(tracking_uri=uri)
    champion = client.get_model_version_by_alias(MODEL_NAME, promote.CHAMPION_ALIAS)
    assert str(champion.version) == second.version


def test_promote_champion_raises_without_candidates(tmp_path: Path) -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("promote-empty")
    with mlflow.start_run() as parent:  # parent with no candidate children
        parent_run_id = parent.info.run_id
    with pytest.raises(RuntimeError, match="no candidate child runs"):
        promote.promote_champion(parent_run_id, MODEL_NAME, tracking_uri=uri)
