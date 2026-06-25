"""Structural tests for the Prefect flow — no live server / no real pipeline run.

Imports the flow module and asserts it cleanly wraps the ``pipeline_tasks`` core
in the correct order, with each Prefect task wired to a real core callable.
Skipped when prefect is not importable; flow validation then runs wherever prefect
is installed (it imports cleanly cross-platform, unlike airflow).

These tests are purely structural — they never call ``clinops_flow()``, so no
Synthea cohort is extracted and no model is trained.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path
from typing import Any

import pytest

from clinops.tasks import pipeline_tasks

pytest.importorskip("prefect")

FLOW_PATH = Path(__file__).resolve().parents[1] / "orchestration" / "prefect" / "clinops_flow.py"
FLOW_NAME = "clinops_pipeline"
# The pipeline order the flow must encode (extract -> ... -> package).
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
def flow_module() -> Any:
    spec = importlib.util.spec_from_file_location("clinops_flow", FLOW_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # raises if the flow has import errors
    return module


def test_flow_imports_without_errors(flow_module: Any) -> None:
    from prefect import Flow

    assert isinstance(flow_module.clinops_flow, Flow)
    assert flow_module.clinops_flow.name == FLOW_NAME
    assert len(flow_module.PREFECT_TASKS) == len(EXPECTED_ORDER)


def test_task_ids_match_pipeline(flow_module: Any) -> None:
    task_ids = [task_id for task_id, _ in flow_module.PREFECT_TASKS]
    assert sorted(task_ids) == sorted(EXPECTED_ORDER)


def test_linear_dependency_order(flow_module: Any) -> None:
    # Prefect encodes the linear graph as the ordered task sequence the flow chains
    # (each step's output feeds the next), mirroring the Airflow DAG's edges.
    task_ids = [task_id for task_id, _ in flow_module.PREFECT_TASKS]
    assert task_ids == EXPECTED_ORDER


def test_each_task_wraps_real_pipeline_callable(flow_module: Any) -> None:
    from prefect.tasks import Task

    for task_id, prefect_task in flow_module.PREFECT_TASKS:
        assert isinstance(prefect_task, Task)
        assert prefect_task.name == task_id
        core_fn = getattr(pipeline_tasks, task_id)
        # The task must wrap the actual single-source-of-truth function, not a
        # flow-local dummy — same proof the Airflow structural test uses.
        assert getattr(prefect_task.fn, "__wrapped_step__", None) is core_fn
        assert isinstance(core_fn, types.FunctionType)
        assert core_fn.__module__ == "clinops.tasks.pipeline_tasks"
