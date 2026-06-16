"""Airflow DAG — primary scheduled batch pipeline for ClinOps.

This DAG ONLY wraps the orchestrator-agnostic core in
`src/clinops/tasks/pipeline_tasks.py`. Each task calls one of the shared step
functions (extract -> parse_fhir -> build_features -> train -> evaluate ->
promote -> package). Do NOT re-implement pipeline logic here — see CLAUDE.md.
"""

from __future__ import annotations

# from clinops.tasks import pipeline_tasks
# Each Airflow task should be a thin wrapper around a pipeline_tasks.* function.


def build_dag() -> object:
    """Construct and return the ClinOps Airflow DAG.

    Wires `pipeline_tasks` step functions into Airflow operators with the
    desired schedule and dependencies. Returns the DAG object Airflow discovers.
    """
    raise NotImplementedError
