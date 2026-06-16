"""Flatten FHIR R4 bundles into a patient-level tabular cohort.

Walks the resources in each Synthea FHIR Bundle and produces one row per patient
with demographics, encounter-derived counts, chronic-condition flags, a couple of
recent observation values, and the binary readmission target.

Resources are associated to a patient by their ``subject``/``patient`` reference.
For the common Synthea case of one Patient per bundle, all resources in the
bundle are attributed to that patient.

**Readmission target (`readmission_30d`)** — see :func:`label_readmission_30d`:
a patient is labelled ``1`` if any inpatient encounter *starts* within 30 days of
a prior inpatient encounter's *end* (encounters ordered by start time); otherwise
``0``. The same rule is documented in ``docs/architecture.md``.

Synthetic data only — never any real PHI.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from datetime import date, datetime
from typing import Any

import pandas as pd

from clinops.config import settings

logger = logging.getLogger(__name__)

# US Core race/ethnicity extension URLs (Synthea emits these when US Core is on).
RACE_URL = "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race"
ETHNICITY_URL = "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity"

# Chronic conditions matched by case-insensitive substring against the
# Condition's text and coding displays. Column name -> match keywords.
CHRONIC_CONDITIONS: dict[str, tuple[str, ...]] = {
    "diabetes": ("diabetes", "diabetic"),
    "hypertension": ("hypertension", "hypertensive"),
    "heart_failure": ("heart failure", "cardiac failure"),
    "copd": ("copd", "chronic obstructive pulmonary"),
}

# Observation matching (LOINC code or display substring).
BMI_LOINC = "39156-5"
BMI_DISPLAY = "body mass index"
SYSTOLIC_BP_LOINC = "8480-6"
SYSTOLIC_BP_DISPLAY = "systolic"

# Encounter class codes (v3-ActCode) treated as inpatient.
INPATIENT_CLASS_CODES = {"IMP", "ACUTE", "NONAC"}

READMISSION_WINDOW_DAYS = 30


def _iter_resources(bundle: dict[str, Any], resource_type: str) -> Iterator[dict[str, Any]]:
    """Yield every resource of ``resource_type`` within a bundle."""
    for entry in bundle.get("entry", []):
        resource = entry.get("resource") or {}
        if resource.get("resourceType") == resource_type:
            yield resource


def _reference_id(reference: str | None) -> str | None:
    """Extract the bare id from a FHIR reference (``Patient/x`` or ``urn:uuid:x``)."""
    if not reference:
        return None
    return reference.split("/")[-1].split(":")[-1]


def _subject_id(resource: dict[str, Any]) -> str | None:
    """Return the referenced subject/patient id of a resource, if any."""
    ref = (resource.get("subject") or resource.get("patient") or {}).get("reference")
    return _reference_id(ref)


def _parse_dt(value: str | None) -> datetime | None:
    """Parse a FHIR dateTime into a naive ``datetime`` (tzinfo dropped), or None."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=None)


def _last_encounter_date(encounters: list[dict[str, Any]]) -> date | None:
    """Return the date of a patient's most recent encounter, or None.

    For each encounter the end time is used, falling back to its start time; the
    latest such moment across all encounters is returned as a ``date``. Encounters
    with no parseable start or end are ignored.

    Args:
        encounters: All Encounter resources for one patient.

    Returns:
        The most recent encounter date, or None if none have a usable timestamp.
    """
    latest: datetime | None = None
    for encounter in encounters:
        period = encounter.get("period") or {}
        when = _parse_dt(period.get("end")) or _parse_dt(period.get("start"))
        if when is None:
            continue
        if latest is None or when > latest:
            latest = when
    return latest.date() if latest is not None else None


def _calculate_age(
    birth_date: str | None, reference: date, patient_id: str | None = None
) -> float | None:
    """Return age in whole years at ``reference``, or None if birthDate is absent.

    ``reference`` is the patient's last-encounter date (see
    :func:`_last_encounter_date`), so age means "age at last encounter". A
    non-negative result is guaranteed: if the computation still yields a negative
    age (e.g. a birthDate after the reference date), it is clamped to ``0`` and a
    WARNING is logged identifying the patient.

    Args:
        birth_date: The patient's FHIR ``birthDate`` (``YYYY-MM-DD`` or longer).
        reference: Date to compute age against (the last-encounter date).
        patient_id: Patient id, used only for the clamp warning message.

    Returns:
        Age in whole years as a float, or None if ``birth_date`` is absent.
    """
    if not birth_date:
        return None
    try:
        born = date.fromisoformat(birth_date[:10])
    except ValueError:
        return None
    years = reference.year - born.year
    if (reference.month, reference.day) < (born.month, born.day):
        years -= 1
    if years < 0:
        logger.warning(
            "Patient %s has negative derived age (%d years) at reference %s; clamping to 0.",
            patient_id,
            years,
            reference.isoformat(),
        )
        years = 0
    return float(years)


def _extension_text(patient: dict[str, Any], url: str) -> str | None:
    """Read a US Core race/ethnicity extension as text (``text`` or ``ombCategory``)."""
    for extension in patient.get("extension", []):
        if extension.get("url") != url:
            continue
        sub_extensions = extension.get("extension", [])
        for sub in sub_extensions:
            if sub.get("url") == "text":
                return sub.get("valueString")
        for sub in sub_extensions:
            if sub.get("url") == "ombCategory":
                return (sub.get("valueCoding") or {}).get("display")
    return None


def _is_inpatient(encounter: dict[str, Any]) -> bool:
    """Return True if an encounter is an inpatient encounter."""
    encounter_class = encounter.get("class") or {}
    if encounter_class.get("code") in INPATIENT_CLASS_CODES:
        return True
    return "inpatient" in (encounter_class.get("display") or "").lower()


def _condition_text(condition: dict[str, Any]) -> str:
    """Return a lowercased blob of a Condition's text and coding displays."""
    code = condition.get("code") or {}
    parts = [code.get("text") or ""]
    for coding in code.get("coding", []):
        parts.append(coding.get("display") or "")
    return " ".join(part for part in parts if part).lower()


def _coding_matches(coding: dict[str, Any], loinc: str, display_substr: str) -> bool:
    """Return True if a coding matches by LOINC code or display substring."""
    if coding.get("code") == loinc:
        return True
    return display_substr in (coding.get("display") or "").lower()


def _observation_value(
    observation: dict[str, Any], loinc: str, display_substr: str
) -> float | None:
    """Extract a numeric value from an Observation or one of its components."""
    code = observation.get("code") or {}
    direct_match = display_substr in (code.get("text") or "").lower() or any(
        _coding_matches(coding, loinc, display_substr) for coding in code.get("coding", [])
    )
    if direct_match:
        value = (observation.get("valueQuantity") or {}).get("value")
        if value is not None:
            return float(value)

    for component in observation.get("component", []):
        comp_code = component.get("code") or {}
        if any(_coding_matches(c, loinc, display_substr) for c in comp_code.get("coding", [])):
            value = (component.get("valueQuantity") or {}).get("value")
            if value is not None:
                return float(value)
    return None


def _most_recent_observation(
    observations: list[dict[str, Any]], loinc: str, display_substr: str
) -> float | None:
    """Return the most recent matching observation value by effective date, or None."""
    best_when: datetime | None = None
    best_value: float | None = None
    for observation in observations:
        value = _observation_value(observation, loinc, display_substr)
        if value is None:
            continue
        when = _parse_dt(observation.get("effectiveDateTime")) or datetime.min
        if best_when is None or when >= best_when:
            best_when = when
            best_value = value
    return best_value


def label_readmission_30d(encounters: list[dict[str, Any]]) -> int:
    """Label 30-day inpatient readmission for a single patient.

    Rule: order the patient's inpatient encounters by start time. The label is
    ``1`` if any encounter *starts* on or within ``READMISSION_WINDOW_DAYS`` (30)
    days after the *end* of the immediately preceding inpatient encounter;
    otherwise ``0``. Encounters missing a start time are ignored; an encounter
    missing an end time falls back to its own start time.

    Args:
        encounters: All Encounter resources for one patient.

    Returns:
        ``1`` if a 30-day inpatient readmission is detected, else ``0``.
    """
    spans: list[tuple[datetime, datetime]] = []
    for encounter in encounters:
        if not _is_inpatient(encounter):
            continue
        period = encounter.get("period") or {}
        start = _parse_dt(period.get("start"))
        if start is None:
            continue
        end = _parse_dt(period.get("end")) or start
        spans.append((start, end))

    spans.sort(key=lambda span: span[0])
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:], strict=False):
        gap_days = (next_start - prev_end).days
        if 0 <= gap_days <= READMISSION_WINDOW_DAYS:
            return 1
    return 0


def _flatten_patient(
    patient: dict[str, Any],
    encounters: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a single flat record for one patient and their resources."""
    patient_id = patient.get("id")
    # Age is measured at the patient's last encounter; the config reference_date
    # is only a fallback for patients with no dated encounters.
    reference = _last_encounter_date(encounters) or settings.reference_date
    record: dict[str, Any] = {
        "patient_id": patient_id,
        "age": _calculate_age(patient.get("birthDate"), reference, patient_id),
        "gender": patient.get("gender"),
        "race": _extension_text(patient, RACE_URL),
        "ethnicity": _extension_text(patient, ETHNICITY_URL),
        "encounter_count": len(encounters),
        "inpatient_count": sum(1 for enc in encounters if _is_inpatient(enc)),
    }

    condition_blobs = [_condition_text(condition) for condition in conditions]
    for name, keywords in CHRONIC_CONDITIONS.items():
        present = any(any(kw in blob for kw in keywords) for blob in condition_blobs)
        record[f"cond_{name}"] = int(present)

    record["bmi"] = _most_recent_observation(observations, BMI_LOINC, BMI_DISPLAY)
    record["systolic_bp"] = _most_recent_observation(
        observations, SYSTOLIC_BP_LOINC, SYSTOLIC_BP_DISPLAY
    )

    record[settings.target_name] = label_readmission_30d(encounters)
    return record


def _flatten_bundle(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every patient in a bundle into a list of flat records."""
    patients = list(_iter_resources(bundle, "Patient"))
    if not patients:
        return []

    encounters = list(_iter_resources(bundle, "Encounter"))
    conditions = list(_iter_resources(bundle, "Condition"))
    observations = list(_iter_resources(bundle, "Observation"))

    single_patient = len(patients) == 1
    records: list[dict[str, Any]] = []
    for patient in patients:
        if single_patient:
            patient_encounters = encounters
            patient_conditions = conditions
            patient_observations = observations
        else:
            patient_id = patient.get("id")
            patient_encounters = [e for e in encounters if _subject_id(e) == patient_id]
            patient_conditions = [c for c in conditions if _subject_id(c) == patient_id]
            patient_observations = [o for o in observations if _subject_id(o) == patient_id]
        records.append(
            _flatten_patient(patient, patient_encounters, patient_conditions, patient_observations)
        )
    return records


def log_class_balance(cohort: pd.DataFrame) -> None:
    """Log the target class balance and warn loudly if it looks degenerate."""
    target = settings.target_name
    if target not in cohort.columns or cohort.empty:
        return

    total = len(cohort)
    positives = int((cohort[target] == 1).sum())
    negatives = total - positives
    pct = 100.0 * positives / total if total else 0.0
    logger.info(
        "Class balance for %s: positives=%d (%.1f%%), negatives=%d, total=%d",
        target,
        positives,
        pct,
        negatives,
        total,
    )
    if positives == 0 or pct < 1.0:
        logger.warning(
            "Label %s looks DEGENERATE (positives=%d, %.2f%%). The 30-day "
            "readmission derivation may need revisiting before training.",
            target,
            positives,
            pct,
        )


def parse_bundles(bundles: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Flatten FHIR R4 bundles into a patient-level cohort DataFrame.

    Args:
        bundles: Iterable of FHIR Bundle dictionaries (from the Synthea loader).

    Returns:
        A DataFrame with one row per patient. Logs the resulting class balance.
    """
    records: list[dict[str, Any]] = []
    for bundle in bundles:
        records.extend(_flatten_bundle(bundle))

    cohort = pd.DataFrame(records)
    if cohort.empty:
        logger.warning("No patient records parsed; cohort is empty.")
    else:
        log_class_balance(cohort)
    return cohort
