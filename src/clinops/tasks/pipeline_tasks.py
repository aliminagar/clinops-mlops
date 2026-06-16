"""The orchestrator-agnostic pipeline core.

This module is the single source of truth for the pipeline. It defines each step
as a plain Python function:

    extract -> parse_fhir -> build_features -> train -> evaluate -> promote -> package

The Airflow DAG, the Prefect flow, and the Kubeflow pipeline all IMPORT these
functions and only wrap them with their own scheduling/execution semantics.
**No orchestrator may re-implement this logic** — change a step here and every
orchestrator inherits it. See CLAUDE.md ("orchestrator-agnostic rule").

Each function below delegates to the relevant `clinops.*` module; steps that are
not yet built remain ``NotImplementedError`` stubs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from clinops.config import settings
from clinops.etl import fhir_parser, synthea_loader
from clinops.features import build_features as feature_builder
from clinops.registry import promote as registry_promote
from clinops.training import train as trainer

logger = logging.getLogger(__name__)


def extract() -> Path:
    """Step 1 — ensure synthetic Synthea FHIR R4 bundles exist in ``data/raw/``.

    Returns:
        The raw data directory containing the FHIR bundles.

    Raises:
        FileNotFoundError: If no bundles are present (points the user at
            ``make synthea`` to generate synthetic data).
    """
    raw_dir = settings.data_raw
    bundle_files = list(raw_dir.rglob("*.json")) if raw_dir.exists() else []
    if not bundle_files:
        raise FileNotFoundError(
            f"No FHIR bundles found under {raw_dir!s}. "
            "Generate synthetic data first with `make synthea`."
        )
    logger.info("Found %d candidate bundle file(s) under %s", len(bundle_files), raw_dir)
    return raw_dir


def parse_fhir(raw_dir: Path | None = None) -> Path:
    """Step 2 — flatten FHIR bundles into ``data/interim/cohort.parquet``.

    Orchestrates the ETL modules only: loads bundles via the Synthea loader,
    flattens them with the FHIR parser, and persists the cohort. No parsing
    logic lives here.

    Args:
        raw_dir: Directory of raw bundles; defaults to ``settings.data_raw``.

    Returns:
        Path to the written cohort parquet file.
    """
    source_dir = Path(raw_dir) if raw_dir is not None else settings.data_raw
    bundles = synthea_loader.iter_bundles(source_dir)
    cohort = fhir_parser.parse_bundles(bundles)

    out_path = settings.data_interim / "cohort.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(out_path, index=False)
    logger.info("Wrote cohort with %d row(s) to %s", len(cohort), out_path)
    return out_path


def build_features(interim_path: Path | None = None) -> Path:
    """Step 3 — build the shared feature store into ``data/processed/``.

    Delegates entirely to :mod:`clinops.features.build_features`; no feature
    logic lives here. Reads the interim cohort and writes the model-agnostic
    feature store (the same ``X``/``y`` consumed by every model).

    Args:
        interim_path: Interim cohort parquet; defaults to
            ``settings.data_interim / "cohort.parquet"``.

    Returns:
        Path to the written feature-store parquet file.
    """
    source = (
        Path(interim_path) if interim_path is not None else settings.data_interim / "cohort.parquet"
    )
    out_path = feature_builder.build_feature_store(source)
    logger.info("Built feature store at %s", out_path)
    return out_path


def train(feature_path: Path | None = None) -> dict[str, Any]:
    """Step 4 — train the baselines, log to MLflow, and register the lead model.

    Delegates to :func:`clinops.training.train.run_training`; no modelling logic
    lives here. The PyTorch challenger is not yet wired in (its module is still a
    stub), so this trains the sklearn + LightGBM baselines only.

    Args:
        feature_path: Feature-store parquet; defaults to the configured path.

    Returns:
        The MLflow references for this run: the tracking URI, experiment, parent
        and per-model run IDs, and the registered (LightGBM) model name/version.

    Raises:
        RuntimeError: If MLflow tracking produced no references (tracking off).
    """
    summary = trainer.run_training(feature_path)
    mlflow_refs = summary.get("mlflow")
    if mlflow_refs is None:
        raise RuntimeError("Training produced no MLflow references; is tracking enabled?")
    return mlflow_refs


def evaluate(run_refs: dict[str, Any] | None = None) -> dict[str, dict[str, float]]:
    """Step 5 — return the per-model comparison metrics from the latest run.

    Training and evaluation are unified in :func:`clinops.training.train.run_training`
    (which scores val + test once, with no leakage); this step surfaces the
    persisted test-fold comparison metrics that back the same MLflow runs.

    Args:
        run_refs: Optional MLflow references from :func:`train` (unused for the
            on-disk read; accepted so orchestrators can pass the upstream output).

    Returns:
        A mapping of model name -> test-fold metric name -> value.
    """
    metrics_path = settings.reports_dir / "metrics.json"
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {name: block["test"] for name, block in summary["models"].items()}


def promote(metrics: Any = None) -> dict[str, Any]:
    """Step 6 — promote the CV-best champion in the MLflow registry.

    Reads the parent run and registered-model name from the persisted training
    summary and delegates to :func:`clinops.registry.promote.promote_champion`;
    no selection logic lives here. Safe and idempotent to run after :func:`train`
    (the same runs always resolve the same champion).

    Args:
        metrics: Upstream evaluate output; accepted for orchestrator wiring but
            unused — the parent run is read from the persisted summary.

    Returns:
        The champion model name, registry version, and deployed F2 threshold.

    Raises:
        RuntimeError: If the training summary has no MLflow references (train was
            run without tracking).
    """
    summary = json.loads((settings.reports_dir / "metrics.json").read_text(encoding="utf-8"))
    mlflow_refs = summary.get("mlflow")
    if mlflow_refs is None:
        raise RuntimeError("Training summary has no MLflow references; run train with tracking.")

    result = registry_promote.promote_champion(
        mlflow_refs["parent_run_id"], mlflow_refs["registered_model"]["name"]
    )
    logger.info("Promoted champion %s -> v%s", result.winner, result.version)
    return {
        "champion_model": result.winner,
        "model_version": result.version,
        "operating_threshold": result.threshold,
    }


def package(champion_version: str | None = None) -> dict[str, Any]:
    """Step 7 — resolve the champion the BentoML service will serve.

    Delegates to :mod:`clinops.serving.bentoml_service` (imported lazily so the
    pipeline core does not require BentoML); no serving logic lives here. Returns
    the serving manifest (model version, deployed F2 threshold, feature schema)
    that the ``/predict`` endpoint exposes. Build the bento itself with
    ``bentoml build`` (see ``bentofile.yaml``).

    Args:
        champion_version: Optional expected champion version (from the promote
            step) to verify against what the registry resolves.

    Returns:
        The serving manifest for the resolved champion.

    Raises:
        RuntimeError: If ``champion_version`` is given but does not match the
            champion the registry resolves.
    """
    from clinops.serving import bentoml_service

    champion = bentoml_service.load_champion()
    if champion_version is not None and str(champion_version) != champion.version:
        raise RuntimeError(
            f"Requested champion version {champion_version!r} but registry resolved "
            f"version {champion.version!r}."
        )
    manifest = bentoml_service.champion_manifest(champion)
    logger.info("Packaged champion v%s for serving", manifest["model_version"])
    return manifest


def run_pipeline() -> Any:
    """Convenience entry point: run the full pipeline end to end in-process.

    Wires the steps together (extract -> ... -> package). Orchestrators may call
    this directly or wrap the individual step functions for finer-grained tasks.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # CLI entry point used by the Makefile (`--step etl|train|...`).
    raise NotImplementedError
