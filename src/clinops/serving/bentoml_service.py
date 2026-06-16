"""BentoML service that serves the MLflow registry champion at ``POST /predict``.

The service loads the champion model from the MLflow Model Registry **by name**
(``settings.registered_model_name``), preferring a ``champion`` alias and falling
back to the latest version — never a hardcoded path. The deployed decision
threshold is read from the champion run's logged ``operating_threshold`` metric
(the F2, recall-weighted screening point), so it is never hardcoded either.

The champion is loaded as a flavor-agnostic ``mlflow.pyfunc`` model, so a logistic
regression, a LightGBM tree, or the PyTorch challenger all serve through the same
path — ``predict`` returns the positive-class probability regardless of flavor
(see :class:`clinops.serving.proba_model.ProbaModel`).

The request payload is the engineered feature schema. The expected feature names
are pulled from the loaded model's input **signature** and the payload is
validated against them, so a train/serve schema skew is rejected at the door. The
response always returns the calibrated probability, the threshold, and the boolean
screening flag (``probability >= threshold``) plus the served model version.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import bentoml
import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from clinops.config import settings

logger = logging.getLogger(__name__)

# Registry alias preferred when resolving the champion; falls back to latest.
CHAMPION_ALIAS = "champion"
# Run metric holding the deployed (F2) operating threshold; see training stage.
OPERATING_THRESHOLD_METRIC = "operating_threshold"


class FeatureValidationError(ValueError):
    """Raised when a request's feature set does not match the model schema."""


class PredictRequest(BaseModel):
    """Request body for ``/predict``: the engineered feature row to score."""

    features: dict[str, float] = Field(
        ...,
        description="Engineered features keyed by name; must match the model schema exactly.",
    )


class PredictResponse(BaseModel):
    """Response body for ``/predict`` — probability, threshold, flag, version."""

    probability: float = Field(..., description="Calibrated positive-class probability.")
    threshold: float = Field(..., description="Deployed F2 screening threshold.")
    readmission_flag: bool = Field(..., description="True iff probability >= threshold.")
    model_version: str = Field(..., description="Served MLflow registry model version.")


@dataclass(frozen=True)
class Champion:
    """A resolved champion ready to serve: model, version, threshold, schema."""

    model: Any
    version: str
    threshold: float
    feature_names: list[str]


def _feature_names(model: Any) -> list[str]:
    """Pull the expected feature names from the pyfunc model's input signature."""
    schema = model.metadata.get_input_schema()
    names = schema.input_names() if schema is not None else None
    if not names:
        raise RuntimeError(
            "Champion model has no input signature; cannot determine the feature schema. "
            "Models must be logged with a signature (see training stage)."
        )
    return list(names)


def _resolve_version(client: MlflowClient, name: str, alias: str) -> Any:
    """Resolve the champion model version by alias, falling back to the latest."""
    try:
        version = client.get_model_version_by_alias(name, alias)
        logger.info("Resolved champion via alias '%s' -> version %s", alias, version.version)
        return version
    except MlflowException:
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            raise RuntimeError(f"No versions registered for model {name!r}.") from None
        version = max(versions, key=lambda mv: int(mv.version))
        logger.info("No '%s' alias for %r; using latest version %s", alias, name, version.version)
        return version


def load_champion(
    name: str | None = None,
    *,
    alias: str = CHAMPION_ALIAS,
    tracking_uri: str | None = None,
) -> Champion:
    """Load the champion model, threshold, and feature schema from the registry.

    Args:
        name: Registered model name; defaults to ``settings.registered_model_name``.
        alias: Registry alias to prefer; falls back to the latest version.
        tracking_uri: MLflow tracking URI; defaults to ``settings.mlflow_tracking_uri``.

    Returns:
        A :class:`Champion` with the fitted model, version, deployed threshold, and
        the ordered feature names the model expects.

    Raises:
        RuntimeError: If no version is registered or the run has no logged threshold.
    """
    # MLflow 3.x gates the local file store; opt in so the ``mlruns/`` default works.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    model_name = name or settings.registered_model_name
    uri = tracking_uri or settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    version = _resolve_version(client, model_name, alias)
    # Flavor-agnostic: the champion may be a tree, a linear model, or the torch
    # challenger — pyfunc serves whichever won, returning positive-class probability.
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version.version}")

    run = client.get_run(version.run_id)
    threshold = run.data.metrics.get(OPERATING_THRESHOLD_METRIC)
    if threshold is None:
        raise RuntimeError(
            f"Run {version.run_id} has no logged {OPERATING_THRESHOLD_METRIC!r}; "
            "cannot determine the deployed operating threshold."
        )

    champion = Champion(
        model=model,
        version=str(version.version),
        threshold=float(threshold),
        feature_names=_feature_names(model),
    )
    logger.info(
        "Loaded champion v%s (threshold=%.4f, %d features)",
        champion.version,
        champion.threshold,
        len(champion.feature_names),
    )
    return champion


def validate_features(features: dict[str, float], expected: list[str]) -> None:
    """Reject a payload whose feature keys differ from the model schema.

    Args:
        features: The provided feature mapping.
        expected: The feature names the model expects.

    Raises:
        FeatureValidationError: If any expected feature is missing or any extra
            (unknown) feature is present.
    """
    provided = set(features)
    expected_set = set(expected)
    missing = sorted(expected_set - provided)
    unknown = sorted(provided - expected_set)
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"missing features {missing}")
        if unknown:
            problems.append(f"unknown features {unknown}")
        raise FeatureValidationError(
            "Feature schema mismatch (no train/serve skew allowed): " + "; ".join(problems)
        )


def champion_manifest(champion: Champion) -> dict[str, Any]:
    """Summarize what the serving layer will expose (no model object).

    Args:
        champion: The loaded champion.

    Returns:
        A JSON-friendly manifest: model name, version, deployed threshold, and the
        ordered feature schema the ``/predict`` endpoint validates against.
    """
    return {
        "registered_model_name": settings.registered_model_name,
        "model_version": champion.version,
        "operating_threshold": champion.threshold,
        "n_features": len(champion.feature_names),
        "feature_names": champion.feature_names,
    }


def predict_one(champion: Champion, features: dict[str, float]) -> PredictResponse:
    """Score a single validated feature row against the champion model.

    Args:
        champion: The loaded champion.
        features: Feature mapping; validated against the champion schema.

    Returns:
        The full prediction response (probability, threshold, flag, version).
    """
    validate_features(features, champion.feature_names)
    row = pd.DataFrame(
        [[float(features[name]) for name in champion.feature_names]],
        columns=champion.feature_names,
    )
    # The pyfunc model returns the positive-class probability directly (see
    # clinops.serving.proba_model.ProbaModel), regardless of the champion flavor.
    probability = float(np.asarray(champion.model.predict(row)).ravel()[0])
    return PredictResponse(
        probability=probability,
        threshold=champion.threshold,
        readmission_flag=probability >= champion.threshold,
        model_version=champion.version,
    )


@bentoml.service(name="clinops_readmission")
class ReadmissionService:
    """Serves the registry champion's 30-day readmission screening prediction."""

    def __init__(self) -> None:
        """Load the champion from the registry once at service startup."""
        self.champion = load_champion()

    @bentoml.api
    def predict(self, request: PredictRequest) -> PredictResponse:
        """Return the calibrated probability and F2 screening flag for one patient."""
        try:
            return predict_one(self.champion, request.features)
        except FeatureValidationError as exc:
            raise bentoml.exceptions.InvalidArgument(str(exc)) from exc


# Alias kept for `bentoml serve clinops.serving.bentoml_service:svc` (see Makefile).
svc = ReadmissionService
