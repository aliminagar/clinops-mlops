"""Build the shared model feature matrix from the parsed FHIR cohort.

Transforms the flattened, patient-level cohort produced by the ETL stage
(``data/interim/cohort.parquet`` — one row per patient, see
:mod:`clinops.etl.fhir_parser`) into the final feature matrix ``X`` and aligned
target vector ``y``. This feature store is **model-agnostic**: the sklearn
baselines and the PyTorch challenger train on the exact same ``X``/``y``, so it
is built once here and persisted to ``data/processed/``.

Leakage control (non-negotiable)
--------------------------------
The label ``readmission_30d`` is defined by a *second* inpatient encounter that
starts within 30 days of a prior one (see
:func:`clinops.etl.fhir_parser.label_readmission_30d`). The raw aggregate counts
in the cohort (``encounter_count``, ``inpatient_count``) therefore *include* that
readmission-defining encounter whenever the label is positive — using them
verbatim would leak the outcome into the features.

To enforce the index-admission cutoff at the aggregate level we subtract the
single readmission-defining encounter from the utilization counts::

    prior_inpatient_count = max(inpatient_count - readmission_30d, 0)
    prior_encounter_count = max(encounter_count - readmission_30d, 0)

The raw counts are **not** emitted. This is leakage *removal*, not leakage: two
patients with identical prior histories get identical features regardless of
whether a readmission later occurred (e.g. a patient with one prior inpatient
stay who is readmitted, ``inpatient_count=2, label=1``, and one who is not,
``inpatient_count=1, label=0``, both yield ``prior_inpatient_count=1``).

Other properties
----------------
- **Deterministic.** No randomness; categorical encoding and median imputation
  are computed only from the data present, and the output columns are sorted, so
  the same input always yields byte-identical ``X``.
- **No NaNs.** Numeric features are median-imputed (with a companion
  ``*_missing`` indicator), so the emitted matrix never contains missing values.
- **Graceful.** Source columns that are absent (e.g. medication or ED-visit
  counts, which the current ETL does not extract) are skipped rather than
  crashing, so the builder degrades cleanly as the cohort schema grows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from clinops.config import settings
from clinops.etl.fhir_parser import log_class_balance

logger = logging.getLogger(__name__)

# Chronic-condition flag columns emitted by the FHIR parser. Each is a 0/1 flag;
# their sum is the patient's distinct chronic-condition burden.
_CONDITION_FLAGS: tuple[str, ...] = (
    "cond_diabetes",
    "cond_hypertension",
    "cond_heart_failure",
    "cond_copd",
)

# Numeric observation columns: median-imputed, each paired with a missing flag.
_OBSERVATION_COLUMNS: tuple[str, ...] = ("bmi", "systolic_bp")

# Categorical demographic columns to one-hot encode. Race/ethnicity are present
# in the cohort but intentionally excluded from features (high-cardinality
# demographic attributes we do not want the model to key on).
_CATEGORICAL_COLUMNS: tuple[str, ...] = ("gender",)


def _one_hot(series: pd.Series, prefix: str) -> pd.DataFrame:
    """One-hot encode a categorical series over its observed categories.

    Categories are taken from the data present and sorted for determinism;
    missing values encode as all-zero rows.

    Args:
        series: The categorical column to encode.
        prefix: Column-name prefix for the generated indicator columns.

    Returns:
        A DataFrame of ``int`` indicator columns, one per observed category.
    """
    categories = sorted(str(value) for value in series.dropna().unique())
    columns = {f"{prefix}_{cat}": (series == cat).astype(int) for cat in categories}
    return pd.DataFrame(columns, index=series.index)


def _impute_median(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Median-impute a numeric series and return a companion missing indicator.

    The median is computed only over present values; if every value is missing
    the column is filled with ``0.0``. The returned indicator is ``1`` where the
    original value was missing.

    Args:
        series: The numeric column to impute.

    Returns:
        A tuple of (imputed float series, ``int`` missing-value indicator).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    missing = numeric.isna().astype(int)
    median = numeric.median()
    fill = float(median) if pd.notna(median) else 0.0
    return numeric.fillna(fill).astype(float), missing


def _leakage_adjusted_count(counts: pd.Series, label: pd.Series) -> pd.Series:
    """Remove the single readmission-defining encounter from a utilization count.

    Subtracts the (0/1) label from the raw count and clamps at zero, yielding the
    pre-readmission count (see module docstring on leakage control).

    Args:
        counts: Raw aggregate count (e.g. ``inpatient_count``).
        label: The binary ``readmission_30d`` target aligned to ``counts``.

    Returns:
        The leakage-adjusted count as an ``int`` series.
    """
    adjusted = pd.to_numeric(counts, errors="coerce").fillna(0) - label
    return adjusted.clip(lower=0).astype(int)


def build_features(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Construct the leakage-safe feature matrix and aligned target vector.

    Engineers patient-level features from the flattened cohort: demographics
    (age, one-hot gender), leakage-adjusted utilization (prior encounter and
    inpatient counts), and clinical burden (chronic-condition flags + their
    count, median-imputed vitals with missing indicators). Absent source columns
    are skipped. The result is deterministic and free of missing values.

    Args:
        records: The patient-level cohort (one row per patient) produced by the
            ETL stage, including the ``readmission_30d`` target column.

    Returns:
        A tuple of (feature matrix ``X``, target vector ``y``). ``X`` is indexed
        by ``patient_id`` (when present) with columns sorted for determinism;
        ``y`` is the aligned, integer-typed target named ``settings.target_name``.

    Raises:
        ValueError: If the target column is missing from ``records``.
    """
    target = settings.target_name
    if target not in records.columns:
        raise ValueError(f"Cohort is missing the target column {target!r}; cannot build features.")

    frame = records.copy()
    if "patient_id" in frame.columns:
        frame = frame.set_index("patient_id")

    y = pd.to_numeric(frame[target], errors="coerce").fillna(0).astype(int)
    y.name = target

    features: dict[str, pd.Series] = {}

    # Demographics.
    if "age" in frame.columns:
        features["age"], features["age_missing"] = _impute_median(frame["age"])

    # Leakage-adjusted utilization counts (raw counts are intentionally dropped).
    if "encounter_count" in frame.columns:
        features["prior_encounter_count"] = _leakage_adjusted_count(frame["encounter_count"], y)
    if "inpatient_count" in frame.columns:
        features["prior_inpatient_count"] = _leakage_adjusted_count(frame["inpatient_count"], y)

    # Clinical burden: chronic-condition flags and their distinct count.
    present_flags = [flag for flag in _CONDITION_FLAGS if flag in frame.columns]
    for flag in present_flags:
        features[flag] = pd.to_numeric(frame[flag], errors="coerce").fillna(0).astype(int)
    if present_flags:
        flags_frame = pd.concat([features[flag] for flag in present_flags], axis=1)
        features["chronic_condition_count"] = flags_frame.sum(axis=1).astype(int)

    # Median-imputed observation values, each with a missing-value indicator.
    for column in _OBSERVATION_COLUMNS:
        if column in frame.columns:
            features[column], features[f"{column}_missing"] = _impute_median(frame[column])

    # One-hot encoded categoricals.
    categorical_frames: list[pd.DataFrame] = [
        _one_hot(frame[column], prefix=column)
        for column in _CATEGORICAL_COLUMNS
        if column in frame.columns
    ]

    feature_frame = pd.DataFrame(features, index=frame.index)
    x = pd.concat([feature_frame, *categorical_frames], axis=1)
    # Sort columns for deterministic output and guarantee no missing values.
    x = x.reindex(sorted(x.columns), axis=1).fillna(0.0)

    logger.info("Built feature matrix: shape=(%d, %d)", x.shape[0], x.shape[1])
    log_class_balance(y.to_frame(name=target))
    return x, y


def build_feature_store(
    interim_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Build the feature store from the interim cohort and persist it to parquet.

    Reads the patient-level cohort, builds ``X``/``y`` via :func:`build_features`,
    and writes a single parquet file holding the feature matrix with the target
    column appended (indexed by ``patient_id``). Downstream stages reload it and
    split off ``settings.target_name`` to recover ``X`` and ``y``.

    Args:
        interim_path: Path to the interim cohort parquet; defaults to
            ``settings.data_interim / "cohort.parquet"``.
        output_path: Destination parquet path; defaults to
            ``settings.data_processed / "features.parquet"``.

    Returns:
        The path to the written feature-store parquet file.
    """
    source = (
        Path(interim_path) if interim_path is not None else settings.data_interim / "cohort.parquet"
    )
    destination = (
        Path(output_path)
        if output_path is not None
        else settings.data_processed / "features.parquet"
    )

    cohort = pd.read_parquet(source)
    x, y = build_features(cohort)

    store = x.copy()
    store[settings.target_name] = y
    destination.parent.mkdir(parents=True, exist_ok=True)
    store.to_parquet(destination)
    logger.info("Wrote feature store with %d row(s) to %s", len(store), destination)
    return destination
