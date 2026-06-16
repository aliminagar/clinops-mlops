"""Tests for the BentoML serving layer.

Self-contained: each test logs a tiny LightGBM model to a temporary MLflow
registry and exercises the champion loader / predictor directly. No live server
and no dependency on the real 5626-patient feature store.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.models
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from clinops.serving import bentoml_service as serving
from clinops.serving.proba_model import ProbaModel

# Includes integer-valued columns (prior_inpatient_count, gender_male) to guard
# the MLflow integer-schema pitfall via the float64 signature.
FEATURES = ["age", "bmi", "prior_inpatient_count", "gender_male"]
MODEL_NAME = "clinops-serving-test"


def _log_and_load(tmp_dir: Path, *, threshold: float, estimator: Any = None) -> serving.Champion:
    """Log + register any estimator via the flavor-agnostic pyfunc wrapper; load it.

    Defaults to a (non-LightGBM) logistic regression to prove the serving loader is
    flavor-agnostic — it never calls ``mlflow.lightgbm.load_model``.
    """
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_dir / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("serving-test")

    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {
            "age": rng.random(60),
            "bmi": rng.random(60),
            "prior_inpatient_count": rng.integers(0, 4, 60),  # int column
            "gender_male": rng.integers(0, 2, 60),  # int 0/1 column
        }
    )
    y = (x["age"] > 0.5).astype(int)
    model = estimator if estimator is not None else LogisticRegression(max_iter=200)
    model.fit(x, y)

    # Float64 signature so integer-valued columns accept float payloads at serve time.
    signature = mlflow.models.infer_signature(x.astype("float64"), model.predict_proba(x)[:, 1])
    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(
            name="model", python_model=ProbaModel(model), signature=signature
        )
        mlflow.log_metric(serving.OPERATING_THRESHOLD_METRIC, threshold)
    mlflow.register_model(info.model_uri, MODEL_NAME)

    return serving.load_champion(MODEL_NAME, tracking_uri=uri)


@pytest.fixture(scope="module")
def champion(tmp_path_factory: pytest.TempPathFactory) -> serving.Champion:
    return _log_and_load(tmp_path_factory.mktemp("registry"), threshold=0.37)


def test_load_champion_pulls_model_threshold_and_schema(champion: serving.Champion) -> None:
    # Feature names come from the model signature, in the trained order.
    assert champion.feature_names == FEATURES
    assert champion.threshold == pytest.approx(0.37)
    assert champion.version == "1"
    # The champion is a flavor-agnostic pyfunc model (not the raw estimator).
    assert hasattr(champion.model, "predict")


def test_predict_one_returns_full_response_shape(champion: serving.Champion) -> None:
    response = serving.predict_one(champion, dict.fromkeys(FEATURES, 0.5))
    assert isinstance(response, serving.PredictResponse)
    assert isinstance(response.probability, float)
    assert 0.0 <= response.probability <= 1.0
    assert isinstance(response.threshold, float)
    assert response.threshold == pytest.approx(0.37)
    assert isinstance(response.readmission_flag, bool)
    assert isinstance(response.model_version, str)
    assert response.model_version == "1"


def test_flag_equals_probability_ge_threshold(champion: serving.Champion) -> None:
    for value in (0.1, 0.5, 0.9):
        response = serving.predict_one(champion, dict.fromkeys(FEATURES, value))
        assert response.readmission_flag is (response.probability >= response.threshold)


def test_served_threshold_is_logged_value_not_a_constant(tmp_path: Path) -> None:
    # An unusual threshold must be read from the run, not hardcoded (0.5 / 0.055).
    champion = _log_and_load(tmp_path, threshold=0.271)
    assert champion.threshold == pytest.approx(0.271)
    assert champion.threshold not in (0.5, 0.055)


def test_missing_feature_is_rejected(champion: serving.Champion) -> None:
    payload = {"age": 1.0, "bmi": 2.0}  # prior_inpatient_count missing
    with pytest.raises(serving.FeatureValidationError, match="missing features"):
        serving.predict_one(champion, payload)


def test_unknown_feature_is_rejected(champion: serving.Champion) -> None:
    payload = {**dict.fromkeys(FEATURES, 0.5), "surprise_feature": 1.0}
    with pytest.raises(serving.FeatureValidationError, match="unknown features"):
        serving.predict_one(champion, payload)


def test_non_lightgbm_flavor_serves(tmp_path: Path) -> None:
    # The default champion is a logistic regression — a non-LightGBM flavor that
    # still serves through the flavor-agnostic pyfunc loader.
    champion = _log_and_load(tmp_path, threshold=0.4)
    response = serving.predict_one(champion, dict.fromkeys(FEATURES, 0.5))
    assert 0.0 <= response.probability <= 1.0


def test_lightgbm_flavor_also_serves(tmp_path: Path) -> None:
    estimator = LGBMClassifier(n_estimators=10, random_state=0, verbose=-1)
    champion = _log_and_load(tmp_path, threshold=0.42, estimator=estimator)
    assert champion.feature_names == FEATURES
    response = serving.predict_one(champion, dict.fromkeys(FEATURES, 0.5))
    assert 0.0 <= response.probability <= 1.0


def test_champion_manifest_shape(champion: serving.Champion) -> None:
    manifest: dict[str, Any] = serving.champion_manifest(champion)
    assert manifest["model_version"] == "1"
    assert manifest["operating_threshold"] == pytest.approx(0.37)
    assert manifest["feature_names"] == FEATURES
    assert manifest["n_features"] == len(FEATURES)
