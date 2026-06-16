"""Airflow DAG — primary scheduled batch pipeline for ClinOps.

This DAG **only wraps** the orchestrator-agnostic core in
`src/clinops/tasks/pipeline_tasks.py`. Each task is a thin wrapper that calls one
of the shared step functions in order:

    extract -> parse_fhir -> build_features -> train -> evaluate -> promote -> package

No pipeline logic lives here — change a step in `pipeline_tasks.py` and every
orchestrator (Airflow/Prefect/Kubeflow) inherits it (see CLAUDE.md).

The DAG is **manual-trigger only** (`schedule=None`, `catchup=False`, no retries).
Step outputs are passed downstream via XCom; `Path` returns are stringified to stay
JSON/XCom-serializable, and the core functions accept those strings transparently.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

from clinops.tasks import pipeline_tasks

DAG_ID = "clinops_pipeline"

# Ordered pipeline steps. Each value is the real single-source-of-truth callable
# in pipeline_tasks that the matching Airflow task wraps — nothing is re-implemented.
PIPELINE_STEPS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("extract", pipeline_tasks.extract),
    ("parse_fhir", pipeline_tasks.parse_fhir),
    ("build_features", pipeline_tasks.build_features),
    ("train", pipeline_tasks.train),
    ("evaluate", pipeline_tasks.evaluate),
    ("promote", pipeline_tasks.promote),
    ("package", pipeline_tasks.package),
)


def _xcom_safe(value: Any) -> Any:
    """Make a step's return value XCom/JSON-serializable (``Path`` -> ``str``)."""
    return str(value) if isinstance(value, Path) else value


def _make_runner(step: Callable[..., Any], upstream_task_id: str | None) -> Callable[..., Any]:
    """Build the thin Airflow callable for one core step — no pipeline logic here.

    The returned callable pulls the single upstream XCom value (if any), calls the
    matching ``pipeline_tasks`` function, and returns an XCom-friendly value. The
    wrapped core function is exposed via ``__wrapped_step__`` for structural tests.

    Args:
        step: The ``pipeline_tasks`` function this task wraps.
        upstream_task_id: Task id whose output feeds this step, or None for the first.

    Returns:
        The Airflow ``python_callable`` for this task.
    """

    def runner(**context: Any) -> Any:
        """Execute the wrapped pipeline step for this Airflow task."""
        if upstream_task_id is None:
            return _xcom_safe(step())
        upstream_value = context["ti"].xcom_pull(task_ids=upstream_task_id)
        return _xcom_safe(step(upstream_value))

    runner.__name__ = f"run_{step.__name__}"
    # Expose the wrapped core callable so structural tests can assert real wiring.
    runner.__wrapped_step__ = step  # type: ignore[attr-defined]
    return runner


def build_dag() -> DAG:
    """Construct the ClinOps Airflow DAG that wraps the pipeline core.

    Builds one ``PythonOperator`` per step, chained into a linear dependency graph
    (extract -> ... -> package). Manual trigger only; no schedule, no catchup.

    Returns:
        The DAG object Airflow discovers.
    """
    with DAG(
        dag_id=DAG_ID,
        description="ClinOps end-to-end pipeline wrapping clinops.tasks.pipeline_tasks.",
        schedule=None,
        start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
        catchup=False,
        default_args={"retries": 0},
        tags=["clinops", "mlops"],
        doc_md=__doc__,
    ) as dag:
        operators: list[PythonOperator] = []
        previous_task_id: str | None = None
        for task_id, step in PIPELINE_STEPS:
            operator = PythonOperator(
                task_id=task_id,
                python_callable=_make_runner(step, previous_task_id),
            )
            if operators:
                operators[-1].set_downstream(operator)
            operators.append(operator)
            previous_task_id = task_id
    return dag


# Module-level DAG object for Airflow/DagBag discovery.
dag = build_dag()
