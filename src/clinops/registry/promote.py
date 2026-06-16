"""Champion/challenger promotion against the MLflow model registry.

This is the **single source of truth** for promotion. The criterion is
cross-validated PR-AUC (``cv_pr_auc_mean``): among the candidate models trained
under one parent run, the highest-CV-PR-AUC model becomes champion. The training
stage and the standalone ``pipeline_tasks.promote`` step both call in here, so
selection logic lives in exactly one place.

:func:`select_champion` is the pure criterion (used in-memory by training).
:func:`promote_champion` re-resolves the candidates from MLflow, registers the
winner, and points the ``champion`` alias at it — deterministic and idempotent:
the same runs always yield the same champion and threshold (a re-run registers a
fresh version of the *same* champion model, with the alias moved to it).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, NamedTuple

from clinops.config import settings

logger = logging.getLogger(__name__)

CHAMPION_ALIAS = "champion"
# MLflow metric keys logged per candidate run by the training stage.
CV_METRIC = "cv_pr_auc_mean"
THRESHOLD_METRIC = "operating_threshold"
# Artifact path each candidate's model is logged under.
MODEL_ARTIFACT_PATH = "model"


class ChampionPromotion(NamedTuple):
    """Outcome of a promotion: winning model, registry version, F2 threshold."""

    winner: str
    version: str
    threshold: float


def select_champion(cv_scores: Mapping[str, float]) -> str:
    """Return the champion model name: the highest cross-validated PR-AUC.

    The criterion lives here so training (in-memory) and :func:`promote_champion`
    (from MLflow) never diverge. Ties resolve deterministically by name.

    Args:
        cv_scores: Mapping of model name -> CV PR-AUC mean.

    Returns:
        The winning model name.

    Raises:
        ValueError: If ``cv_scores`` is empty.
    """
    if not cv_scores:
        raise ValueError("Cannot select a champion from an empty score mapping.")
    return max(sorted(cv_scores), key=lambda name: cv_scores[name])


def promote_champion(
    parent_run_id: str,
    model_name: str,
    *,
    alias: str = CHAMPION_ALIAS,
    tracking_uri: str | None = None,
) -> ChampionPromotion:
    """Promote the CV-best candidate under a parent run to registry champion.

    Re-resolves the per-model child runs nested under ``parent_run_id``, selects
    the highest-CV-PR-AUC one (:func:`select_champion`), registers its logged model
    under ``model_name``, and points ``alias`` at the new version. Runnable
    standalone after training, and idempotent: the same runs yield the same winner
    and threshold (a re-run registers a fresh version of the same champion model).

    Args:
        parent_run_id: Parent MLflow run grouping the candidate child runs.
        model_name: Registered model name to promote the champion under.
        alias: Registry alias to point at the champion (default ``champion``).
        tracking_uri: MLflow tracking URI; defaults to ``settings.mlflow_tracking_uri``.

    Returns:
        A :class:`ChampionPromotion` (winner name, registry version, F2 threshold).

    Raises:
        RuntimeError: If the parent run has no candidate child runs with a CV score.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = tracking_uri or settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    parent = client.get_run(parent_run_id)
    children = client.search_runs(
        experiment_ids=[parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )
    candidates: dict[str, Any] = {
        run.data.tags.get("mlflow.runName", run.info.run_id): run
        for run in children
        if CV_METRIC in run.data.metrics
    }
    if not candidates:
        raise RuntimeError(f"Parent run {parent_run_id} has no candidate child runs to promote.")

    cv_scores = {name: run.data.metrics[CV_METRIC] for name, run in candidates.items()}
    winner_name = select_champion(cv_scores)
    winner_run = candidates[winner_name]
    threshold = float(winner_run.data.metrics[THRESHOLD_METRIC])

    model_uri = f"runs:/{winner_run.info.run_id}/{MODEL_ARTIFACT_PATH}"
    version = mlflow.register_model(model_uri, model_name)
    client.set_registered_model_alias(model_name, alias, version.version)
    logger.info(
        "Promoted champion %s (CV PR-AUC=%.4f) -> %s v%s (alias %r)",
        winner_name,
        cv_scores[winner_name],
        model_name,
        version.version,
        alias,
    )
    return ChampionPromotion(winner=winner_name, version=str(version.version), threshold=threshold)
