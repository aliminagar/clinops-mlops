"""Structural tests for the Airflow DAG — no live cluster required.

Loads the DAG via ``DagBag`` and asserts it cleanly wraps the ``pipeline_tasks``
core in the correct order, with each task wired to a real core callable. Skipped
when apache-airflow is not importable (e.g. Windows-native, where Airflow cannot
run); DAG validation then runs inside the airflow container / CI.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from clinops.tasks import pipeline_tasks

pytest.importorskip("airflow")

DAGS_DIR = Path(__file__).resolve().parents[1] / "orchestration" / "airflow" / "dags"
DAG_ID = "clinops_pipeline"
# The pipeline order the DAG must encode (extract -> ... -> package).
EXPECTED_ORDER = [
    "extract",
    "parse_fhir",
    "build_features",
    "train",
    "evaluate",
    "promote",
    "package",
]


@pytest.fixture(scope="module")
def dag() -> Any:
    from airflow.models import DagBag

    bag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)
    assert bag.import_errors == {}, f"DAG import errors: {bag.import_errors}"
    loaded = bag.get_dag(DAG_ID)
    assert loaded is not None, f"DAG {DAG_ID!r} not found in {DAGS_DIR}"
    return loaded


def test_dag_imports_without_errors(dag: Any) -> None:
    assert dag.dag_id == DAG_ID
    assert dag.catchup is False
    assert len(dag.tasks) == len(EXPECTED_ORDER)


def test_task_ids_match_pipeline(dag: Any) -> None:
    assert sorted(task.task_id for task in dag.tasks) == sorted(EXPECTED_ORDER)


def test_linear_dependency_order(dag: Any) -> None:
    for index, task_id in enumerate(EXPECTED_ORDER):
        task = dag.get_task(task_id)
        expected_upstream = {EXPECTED_ORDER[index - 1]} if index > 0 else set()
        expected_downstream = (
            {EXPECTED_ORDER[index + 1]} if index < len(EXPECTED_ORDER) - 1 else set()
        )
        assert task.upstream_task_ids == expected_upstream
        assert task.downstream_task_ids == expected_downstream


def test_each_task_wraps_real_pipeline_callable(dag: Any) -> None:
    for task_id in EXPECTED_ORDER:
        task = dag.get_task(task_id)
        core_fn = getattr(pipeline_tasks, task_id)
        # The task must wrap the actual single-source-of-truth function, not a
        # DAG-local dummy. (Some core steps, e.g. promote, are still
        # NotImplementedError pending build — the *wiring* is real regardless.)
        assert getattr(task.python_callable, "__wrapped_step__", None) is core_fn
        assert isinstance(core_fn, types.FunctionType)
        assert core_fn.__module__ == "clinops.tasks.pipeline_tasks"
