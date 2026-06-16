"""Central configuration for ClinOps.

Loads settings from environment variables / a `.env` file (see `.env.example`)
using `pydantic-settings`. Exposes a single, importable `settings` instance so
no other module hardcodes paths, the target name, the random seed, or the age
reference date.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings sourced from the environment / `.env`.

    Attributes:
        data_raw: Directory holding raw Synthea FHIR R4 bundles.
        data_interim: Directory for intermediate, flattened tables.
        data_processed: Directory for the model-ready feature store.
        models_dir: Directory for persisted, fitted model artifacts.
        reports_dir: Directory for metrics JSON and diagnostic plots.
        target_name: Name of the binary prediction target column.
        random_seed: Global seed for reproducible splits / training.
        reference_date: Fallback date for deriving patient age from birthDate,
            used only when a patient has no dated encounters. Age is normally
            measured at the patient's last-encounter date (see
            ``etl.fhir_parser._calculate_age``), not against this fixed date.
        mlflow_tracking_uri: MLflow tracking store; defaults to a local
            ``mlruns/`` directory so runs are captured without a server.
        mlflow_experiment_name: MLflow experiment grouping the training runs.
        registered_model_name: Name of the lead model in the MLflow Model
            Registry (the LightGBM baseline is registered under this name).
        threshold_beta: Beta for the deployed F-beta operating point. Default
            ``2.0`` (F2): for 30-day readmission *screening*, a false negative
            (a missed readmission) is costlier than a false positive (an extra
            follow-up), so the deployed threshold is recall-weighted. The F1
            (``beta=1``) and fixed-0.5 points are still reported alongside; set
            this to ``1.0`` to deploy at F1 instead.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_raw: Path = Path("data/raw")
    data_interim: Path = Path("data/interim")
    data_processed: Path = Path("data/processed")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    target_name: str = "readmission_30d"
    random_seed: int = 42
    # Fallback only — used when a patient has no dated encounters (see docstring).
    reference_date: date = date(2024, 1, 1)

    # --- MLflow tracking / registry spine ---
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "clinops-readmission"
    registered_model_name: str = "clinops-readmission-classifier"

    # --- Operating-point selection ---
    # F2 by default: recall-weighted, since a missed 30-day readmission costs
    # more than a false alarm in a screening setting (see docstring).
    threshold_beta: float = 2.0


# Process-wide settings instance other modules import.
settings = Settings()
