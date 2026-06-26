"""Structural tests for the Kubeflow (KFP) pipeline — no cluster, no kind, no submit.

Imports the pipeline module, compiles it to a temp IR spec, and asserts it cleanly
wraps the ``pipeline_tasks`` core in the correct order, with each component wired to
a real core callable. Skipped when kfp is not importable; compile + structural
validation then run wherever kfp is installed (and in CI if kfp is in
requirements.txt).

These tests never submit the pipeline and never run a step — KFP components execute
in containers on a cluster, which this pass deliberately does not stand up.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path
from typing import Any

import pytest

from clinops.tasks import pipeline_tasks

pytest.importorskip("kfp")

PIPELINE_PATH = Path(__file__).resolve().parents[1] / "orchestration" / "kubeflow" / "pipeline.py"
PIPELINE_NAME = "clinops-pipeline"
# The pipeline order the KFP pipeline must encode (extract -> ... -> package).
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
def pipeline_module() -> Any:
    spec = importlib.util.spec_from_file_location("clinops_kfp_pipeline", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # raises if the pipeline has import errors
    return module


def test_pipeline_imports_without_errors(pipeline_module: Any) -> None:
    assert pipeline_module.PIPELINE_NAME == PIPELINE_NAME
    assert callable(pipeline_module.clinops_pipeline)
    assert len(pipeline_module.PIPELINE_COMPONENTS) == len(EXPECTED_ORDER)


def test_step_ids_match_pipeline(pipeline_module: Any) -> None:
    task_ids = [task_id for task_id, _component, _fn in pipeline_module.PIPELINE_COMPONENTS]
    assert sorted(task_ids) == sorted(EXPECTED_ORDER)


def test_linear_dependency_order(pipeline_module: Any) -> None:
    # KFP encodes the linear graph as task.output -> next-component inputs; the
    # ordered component sequence the pipeline chains mirrors that, parallel to the
    # Airflow DAG's edges and the Prefect flow's task order.
    task_ids = [task_id for task_id, _component, _fn in pipeline_module.PIPELINE_COMPONENTS]
    assert task_ids == EXPECTED_ORDER


def test_each_component_wraps_real_pipeline_callable(pipeline_module: Any) -> None:
    for task_id, component, _fn in pipeline_module.PIPELINE_COMPONENTS:
        core_fn = getattr(pipeline_tasks, task_id)
        # The component must wrap the actual single-source-of-truth function, not a
        # pipeline-local dummy — same proof the Airflow/Prefect tests use.
        assert getattr(component.python_func, "__wrapped_step__", None) is core_fn
        assert isinstance(core_fn, types.FunctionType)
        assert core_fn.__module__ == "clinops.tasks.pipeline_tasks"


def test_pipeline_compiles_to_ir(pipeline_module: Any, tmp_path: Path) -> None:
    out = tmp_path / "clinops_pipeline.yaml"
    returned = pipeline_module.compile_pipeline(str(out))
    assert Path(returned) == out
    assert out.exists() and out.stat().st_size > 0

    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load(out.read_text(encoding="utf-8"))
    # The compiled IR's root DAG must contain exactly the seven steps.
    tasks = spec["root"]["dag"]["tasks"]
    assert len(tasks) == len(EXPECTED_ORDER)
