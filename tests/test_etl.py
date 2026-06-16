"""Tests for the ETL layer (Synthea loader + FHIR parser).

Uses a tiny committed, fully synthetic fixture (no PHI):
- ``tests/fixtures/sample_bundle.json`` — one Bundle with two patients: a 30-day
  inpatient readmission case (patient-a) and a non-readmission case (patient-b).
- ``tests/fixtures/non_patient_bundle.json`` — a Bundle with no Patient resource,
  used to confirm the loader skips it.
"""

from __future__ import annotations

import json
from pathlib import Path

from clinops.config import settings
from clinops.etl import fhir_parser, synthea_loader

FIXTURES = Path(__file__).parent / "fixtures"
TARGET = settings.target_name


def _load_sample_bundle() -> dict:
    with (FIXTURES / "sample_bundle.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _patient_bundle(patient: dict, *resources: dict) -> dict:
    """Wrap a Patient and any related resources in a minimal FHIR collection bundle."""
    entries = [{"resource": patient}, *({"resource": r} for r in resources)]
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def test_loader_yields_only_patient_bundles() -> None:
    bundles = list(synthea_loader.iter_bundles(FIXTURES))
    # Only sample_bundle.json contains a Patient; non_patient_bundle.json is skipped.
    assert len(bundles) == 1
    patient_types = [
        entry["resource"]["resourceType"]
        for entry in bundles[0]["entry"]
        if entry["resource"]["resourceType"] == "Patient"
    ]
    assert patient_types  # at least one Patient present


def test_parser_columns_and_age_derivation() -> None:
    cohort = fhir_parser.parse_bundles([_load_sample_bundle()])

    expected_columns = {
        "patient_id",
        "age",
        "gender",
        "race",
        "ethnicity",
        "encounter_count",
        "inpatient_count",
        "cond_diabetes",
        "cond_hypertension",
        "cond_heart_failure",
        "cond_copd",
        "bmi",
        "systolic_bp",
        TARGET,
    }
    assert expected_columns.issubset(set(cohort.columns))
    assert len(cohort) == 2

    by_id = cohort.set_index("patient_id")
    # Age is derived at each patient's last-encounter date, not a fixed date.
    # patient-a: born 1950-05-15, last encounter ends 2020-01-25 -> 69.
    # patient-b: born 1990-03-20, last encounter ends 2019-06-20 -> 29.
    assert by_id.loc["patient-a", "age"] == 69
    assert by_id.loc["patient-b", "age"] == 29

    assert by_id.loc["patient-a", "gender"] == "male"
    assert by_id.loc["patient-a", "race"] == "White"
    assert by_id.loc["patient-a", "inpatient_count"] == 2
    assert by_id.loc["patient-a", "cond_diabetes"] == 1
    assert by_id.loc["patient-a", "bmi"] == 31.2
    assert by_id.loc["patient-a", "systolic_bp"] == 140  # from the BP panel component

    assert by_id.loc["patient-b", "inpatient_count"] == 1
    assert by_id.loc["patient-b", "cond_hypertension"] == 1
    assert by_id.loc["patient-b", "cond_diabetes"] == 0


def test_label_readmission_30d_via_cohort() -> None:
    cohort = fhir_parser.parse_bundles([_load_sample_bundle()]).set_index("patient_id")
    assert cohort.loc["patient-a", TARGET] == 1  # readmitted within 30 days
    assert cohort.loc["patient-b", TARGET] == 0  # only one inpatient stay


def test_label_readmission_30d_function() -> None:
    readmitted = [
        {
            "class": {"code": "IMP"},
            "period": {
                "start": "2020-01-01T00:00:00+00:00",
                "end": "2020-01-10T00:00:00+00:00",
            },
        },
        {
            "class": {"code": "IMP"},
            "period": {
                "start": "2020-01-20T00:00:00+00:00",
                "end": "2020-01-25T00:00:00+00:00",
            },
        },
    ]
    assert fhir_parser.label_readmission_30d(readmitted) == 1

    not_readmitted = [
        {
            "class": {"code": "IMP"},
            "period": {
                "start": "2019-06-01T00:00:00+00:00",
                "end": "2019-06-05T00:00:00+00:00",
            },
        },
        # A later inpatient stay, but more than 30 days after the first ends.
        {
            "class": {"code": "IMP"},
            "period": {
                "start": "2019-08-01T00:00:00+00:00",
                "end": "2019-08-05T00:00:00+00:00",
            },
        },
    ]
    assert fhir_parser.label_readmission_30d(not_readmitted) == 0


def test_age_uses_last_encounter_not_fixed_reference_date() -> None:
    # Born well after the old fixed reference_date (2024-01-01): a fixed-date
    # computation would yield a negative age. Age must instead be derived from the
    # last encounter (ends 2026-05-01), giving a sensible non-negative value.
    patient = {"resourceType": "Patient", "id": "newborn", "birthDate": "2025-01-01"}
    encounter = {
        "resourceType": "Encounter",
        "class": {"code": "AMB"},
        "subject": {"reference": "Patient/newborn"},
        "period": {
            "start": "2026-05-01T00:00:00+00:00",
            "end": "2026-05-01T01:00:00+00:00",
        },
    }
    cohort = fhir_parser.parse_bundles([_patient_bundle(patient, encounter)])
    age = cohort.set_index("patient_id").loc["newborn", "age"]
    assert age == 1  # 2025-01-01 -> 2026-05-01
    assert age >= 0


def test_age_falls_back_to_reference_date_without_encounter_dates() -> None:
    # The patient has an encounter, but it carries no usable start/end. Age must
    # fall back to the config reference_date without crashing.
    patient = {"resourceType": "Patient", "id": "no-dates", "birthDate": "2000-01-01"}
    undated_encounter = {
        "resourceType": "Encounter",
        "class": {"code": "AMB"},
        "subject": {"reference": "Patient/no-dates"},
    }
    cohort = fhir_parser.parse_bundles([_patient_bundle(patient, undated_encounter)])
    age = cohort.set_index("patient_id").loc["no-dates", "age"]
    # Born 2000-01-01, fallback reference 2024-01-01 -> 24 (no birthday adjustment).
    assert age == settings.reference_date.year - 2000


def test_no_patient_has_negative_age_in_mixed_cohort() -> None:
    # Mixed cohort: the dated fixture patients plus a patient who would be negative
    # under the old fixed-date logic. None may end up with a negative age.
    newborn = {"resourceType": "Patient", "id": "newborn", "birthDate": "2025-06-01"}
    encounter = {
        "resourceType": "Encounter",
        "class": {"code": "AMB"},
        "subject": {"reference": "Patient/newborn"},
        "period": {"end": "2026-01-01T00:00:00+00:00"},
    }
    cohort = fhir_parser.parse_bundles([_load_sample_bundle(), _patient_bundle(newborn, encounter)])
    assert len(cohort) == 3
    assert (cohort["age"].dropna() >= 0).all()
