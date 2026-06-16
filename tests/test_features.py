"""Tests for the feature-engineering layer.

Uses a tiny in-memory synthetic cohort (a few patients, mixed labels) mirroring
the interim schema written by the ETL stage — no real Synthea data required.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clinops.config import settings
from clinops.features import build_features

TARGET = settings.target_name


def _sample_cohort() -> pd.DataFrame:
    """A small patient-level cohort matching the ETL interim schema.

    - patient-a: readmitted (2 inpatient stays, label 1). Its second inpatient
      stay is the labeling encounter.
    - patient-b: a single inpatient stay, never readmitted (label 0) — the
      leakage control must make its prior-inpatient count match patient-a's.
    - patient-c: outpatient only, missing vitals, label 0.
    """
    return pd.DataFrame(
        [
            {
                "patient_id": "patient-a",
                "age": 73.0,
                "gender": "male",
                "race": "White",
                "ethnicity": "Non-Hispanic",
                "encounter_count": 5,
                "inpatient_count": 2,
                "cond_diabetes": 1,
                "cond_hypertension": 1,
                "cond_heart_failure": 0,
                "cond_copd": 0,
                "bmi": 31.2,
                "systolic_bp": 140.0,
                TARGET: 1,
            },
            {
                "patient_id": "patient-b",
                "age": 33.0,
                "gender": "female",
                "race": "Black",
                "ethnicity": "Non-Hispanic",
                "encounter_count": 4,
                "inpatient_count": 1,
                "cond_diabetes": 0,
                "cond_hypertension": 1,
                "cond_heart_failure": 0,
                "cond_copd": 0,
                "bmi": 24.5,
                "systolic_bp": 120.0,
                TARGET: 0,
            },
            {
                "patient_id": "patient-c",
                "age": None,
                "gender": "female",
                "race": "Asian",
                "ethnicity": "Hispanic",
                "encounter_count": 2,
                "inpatient_count": 0,
                "cond_diabetes": 0,
                "cond_hypertension": 0,
                "cond_heart_failure": 0,
                "cond_copd": 0,
                "bmi": None,
                "systolic_bp": None,
                TARGET: 0,
            },
        ]
    )


def test_x_and_y_are_aligned() -> None:
    x, y = build_features.build_features(_sample_cohort())
    assert len(x) == len(y) == 3
    assert list(x.index) == list(y.index) == ["patient-a", "patient-b", "patient-c"]
    assert y.name == TARGET
    assert y.tolist() == [1, 0, 0]


def test_no_nans_in_feature_matrix() -> None:
    x, _ = build_features.build_features(_sample_cohort())
    # Missing age/bmi/systolic_bp for patient-c must be imputed, never left NaN.
    assert not x.isna().any().any()


def test_missing_values_imputed_with_indicator() -> None:
    x, _ = build_features.build_features(_sample_cohort())
    # patient-c has no age/bmi/bp: indicators flag the imputation.
    assert x.loc["patient-c", "age_missing"] == 1
    assert x.loc["patient-c", "bmi_missing"] == 1
    assert x.loc["patient-c", "systolic_bp_missing"] == 1
    assert x.loc["patient-a", "bmi_missing"] == 0
    # Imputed value is the median of present values (73.0, 33.0 -> 53.0).
    assert x.loc["patient-c", "age"] == 53.0


def test_target_is_not_a_feature() -> None:
    x, _ = build_features.build_features(_sample_cohort())
    assert TARGET not in x.columns


def test_leakage_guard_excludes_labeling_encounter() -> None:
    """The readmission-defining encounter must not be counted as a feature.

    patient-a (readmitted: inpatient_count=2, label=1) and patient-b
    (not readmitted: inpatient_count=1, label=0) have the *same* prior inpatient
    history of one stay. After removing the labeling encounter both must yield an
    identical leakage-safe utilization feature, and the raw count must be absent.
    """
    x, _ = build_features.build_features(_sample_cohort())

    # Raw, leaky aggregate counts are not emitted.
    assert "inpatient_count" not in x.columns
    assert "encounter_count" not in x.columns

    # The labeling encounter is removed: 2 - 1 == 1 for the readmitted patient.
    a_prior = x.loc["patient-a", "prior_inpatient_count"]
    b_prior = x.loc["patient-b", "prior_inpatient_count"]
    assert a_prior == 1
    assert b_prior == 1
    assert a_prior == b_prior

    # patient-a's extra (readmission) encounter is likewise not counted.
    assert x.loc["patient-a", "prior_encounter_count"] == 4  # 5 - 1
    assert x.loc["patient-b", "prior_encounter_count"] == 4  # 4 - 0


def test_categorical_and_clinical_burden_features() -> None:
    x, _ = build_features.build_features(_sample_cohort())
    # Deterministic one-hot over observed genders.
    assert x.loc["patient-a", "gender_male"] == 1
    assert x.loc["patient-a", "gender_female"] == 0
    assert x.loc["patient-b", "gender_female"] == 1
    # Distinct chronic-condition burden = sum of cond_* flags.
    assert x.loc["patient-a", "chronic_condition_count"] == 2
    assert x.loc["patient-c", "chronic_condition_count"] == 0


def test_build_is_deterministic() -> None:
    cohort = _sample_cohort()
    x1, y1 = build_features.build_features(cohort)
    x2, y2 = build_features.build_features(cohort)
    pd.testing.assert_frame_equal(x1, x2)
    pd.testing.assert_series_equal(y1, y2)
    # Column order is stable (sorted) across builds.
    assert list(x1.columns) == sorted(x1.columns)


def test_missing_source_columns_are_skipped() -> None:
    # A cohort lacking observation/medication columns must not crash.
    minimal = _sample_cohort()[["patient_id", "age", "gender", "inpatient_count", TARGET]]
    x, y = build_features.build_features(minimal)
    assert len(x) == len(y) == 3
    assert "bmi" not in x.columns
    assert "prior_inpatient_count" in x.columns
    assert not x.isna().any().any()


def test_feature_store_round_trip(tmp_path: Path) -> None:
    interim = tmp_path / "cohort.parquet"
    _sample_cohort().to_parquet(interim, index=False)
    out = tmp_path / "features.parquet"

    written = build_features.build_feature_store(interim, out)
    assert written == out

    store = pd.read_parquet(out)
    assert TARGET in store.columns
    assert len(store) == 3
    # The persisted store reproduces the in-memory build.
    x, y = build_features.build_features(_sample_cohort())
    pd.testing.assert_series_equal(store[TARGET], y, check_names=False)
    assert set(x.columns) == set(c for c in store.columns if c != TARGET)
