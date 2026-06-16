"""Load synthetic Synthea FHIR R4 bundles from the raw data directory.

Discovers the FHIR JSON files Synthea emits (recursively, since Synthea writes to
a `fhir/` subdirectory) and yields the parsed FHIR ``Bundle`` objects. Synthea
also emits ``hospitalInformation*`` and ``practitionerInformation*`` bundles that
contain no ``Patient`` resource; those are skipped — only bundles containing at
least one ``Patient`` are yielded. Malformed or unreadable files are logged and
skipped rather than crashing the run. Synthetic data only — never any real PHI.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _has_patient(bundle: dict[str, Any]) -> bool:
    """Return True if the bundle contains at least one Patient resource."""
    for entry in bundle.get("entry", []):
        resource = entry.get("resource") or {}
        if resource.get("resourceType") == "Patient":
            return True
    return False


def iter_bundles(raw_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield each parsed FHIR Bundle (containing a Patient) found under ``raw_dir``.

    Searches recursively for ``*.json`` files. Non-Bundle files, bundles without a
    Patient resource (e.g. Synthea's hospital/practitioner information bundles),
    and malformed/unreadable files are skipped with a log message.

    Args:
        raw_dir: Directory containing Synthea-generated FHIR bundle files.

    Yields:
        Parsed FHIR Bundle dictionaries that contain a Patient resource.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        logger.warning("Raw data directory does not exist: %s", raw_dir)
        return

    for path in sorted(raw_dir.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                bundle = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable/malformed file %s: %s", path, exc)
            continue

        if not isinstance(bundle, dict) or bundle.get("resourceType") != "Bundle":
            logger.debug("Skipping non-Bundle file: %s", path)
            continue

        if not _has_patient(bundle):
            logger.debug("Skipping bundle without a Patient resource: %s", path)
            continue

        yield bundle
