"""Kubeflow Pipelines (KFP v2) definition for ClinOps.

This pipeline **only wraps** the orchestrator-agnostic core in
`src/clinops/tasks/pipeline_tasks.py`. Each KFP component is a thin wrapper that
calls one of the shared step functions in order:

    extract -> parse_fhir -> build_features -> train -> evaluate -> promote -> package

No pipeline logic lives here — change a step in `pipeline_tasks.py` and every
orchestrator (Airflow/Prefect/Kubeflow) inherits it (see CLAUDE.md).

How this DIFFERS from the Airflow DAG and Prefect flow (and why that matters for
the comparison doc):

- Airflow and Prefect run steps **in-process** and hand Python objects (or XCom
  values) directly from one step to the next.
- KFP components run as **containerized steps** on Kubernetes and pass data across
  container boundaries as **parameters / file artifacts**, not in-process returns.
  So each component here is a self-contained (hermetic) function that imports
  `pipeline_tasks` *inside* the container and returns a KFP parameter. ``Path``
  returns are stringified — which fits, since the core functions already accept the
  ``str`` paths the Airflow XCom marshaling produces — and downstream components
  take that string/dict as a typed input. The wiring (`task.output` -> next
  component) is what KFP turns into the artifact/parameter dependency graph.

Each component delegates to the real single-source-of-truth callable; the wrapped
core function is exposed via ``__wrapped_step__`` (on the component's
``python_func``) so structural tests can assert real wiring, exactly as the
Airflow and Prefect tests do.

Running this for real requires a container image that has ``clinops`` (and its
deps) installed — see the README. This module compiles to an IR spec locally
(``kfp compile``) without a cluster; it is **not** submitted here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kfp import compiler, dsl

from clinops.tasks import pipeline_tasks

PIPELINE_NAME = "clinops-pipeline"
# Base image the components run in. A real cluster run needs an image with
# `clinops` + deps installed (or `packages_to_install=[...]`); compilation does
# not execute the components, so any valid image compiles.
BASE_IMAGE = "python:3.11"


@dsl.component(base_image=BASE_IMAGE)
def extract_op() -> str:
    """Step 1 — wrap ``pipeline_tasks.extract`` (runs containerized)."""
    from clinops.tasks import pipeline_tasks

    return str(pipeline_tasks.extract())


@dsl.component(base_image=BASE_IMAGE)
def parse_fhir_op(raw_dir: str) -> str:
    """Step 2 — wrap ``pipeline_tasks.parse_fhir`` (raw_dir artifact -> interim)."""
    from clinops.tasks import pipeline_tasks

    return str(pipeline_tasks.parse_fhir(raw_dir))


@dsl.component(base_image=BASE_IMAGE)
def build_features_op(interim_path: str) -> str:
    """Step 3 — wrap ``pipeline_tasks.build_features`` (interim -> feature store)."""
    from clinops.tasks import pipeline_tasks

    return str(pipeline_tasks.build_features(interim_path))


@dsl.component(base_image=BASE_IMAGE)
def train_op(feature_path: str) -> dict:
    """Step 4 — wrap ``pipeline_tasks.train`` (feature store -> run refs)."""
    from clinops.tasks import pipeline_tasks

    return pipeline_tasks.train(feature_path)


@dsl.component(base_image=BASE_IMAGE)
def evaluate_op(run_refs: dict) -> dict:
    """Step 5 — wrap ``pipeline_tasks.evaluate`` (run refs -> metrics)."""
    from clinops.tasks import pipeline_tasks

    return pipeline_tasks.evaluate(run_refs)


@dsl.component(base_image=BASE_IMAGE)
def promote_op(metrics: dict) -> dict:
    """Step 6 — wrap ``pipeline_tasks.promote`` (metrics -> champion refs)."""
    from clinops.tasks import pipeline_tasks

    return pipeline_tasks.promote(metrics)


@dsl.component(base_image=BASE_IMAGE)
def package_op(champion: dict) -> dict:
    """Step 7 — wrap ``pipeline_tasks.package`` (champion refs -> serving manifest)."""
    from clinops.tasks import pipeline_tasks

    return pipeline_tasks.package(champion)


# Ordered pipeline steps: (task id, KFP component, real core callable). The core
# callable is the single source of truth the matching component wraps — nothing is
# re-implemented. Mirrors the Airflow DAG's / Prefect flow's PIPELINE_STEPS.
PIPELINE_COMPONENTS: tuple[tuple[str, Any, Callable[..., Any]], ...] = (
    ("extract", extract_op, pipeline_tasks.extract),
    ("parse_fhir", parse_fhir_op, pipeline_tasks.parse_fhir),
    ("build_features", build_features_op, pipeline_tasks.build_features),
    ("train", train_op, pipeline_tasks.train),
    ("evaluate", evaluate_op, pipeline_tasks.evaluate),
    ("promote", promote_op, pipeline_tasks.promote),
    ("package", package_op, pipeline_tasks.package),
)

# Expose the wrapped core callable on each component's python_func so structural
# tests can assert real wiring (test-only handle, not used at run time) — the same
# __wrapped_step__ proof technique the Airflow and Prefect orchestrators use.
for _task_id, _component, _core_fn in PIPELINE_COMPONENTS:
    _component.python_func.__wrapped_step__ = _core_fn


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="ClinOps end-to-end pipeline wrapping pipeline_tasks.",
)
def clinops_pipeline() -> None:
    """Wire the seven ClinOps components into a linear KFP pipeline.

    Each component's single output feeds the next (extract -> ... -> package); KFP
    turns those ``task.output`` references into the cross-container parameter /
    artifact dependency graph. No pipeline logic lives here.
    """
    extract_task = extract_op()
    parse_task = parse_fhir_op(raw_dir=extract_task.output)
    features_task = build_features_op(interim_path=parse_task.output)
    train_task = train_op(feature_path=features_task.output)
    evaluate_task = evaluate_op(run_refs=train_task.output)
    promote_task = promote_op(metrics=evaluate_task.output)
    package_op(champion=promote_task.output)


def compile_pipeline(output_path: str) -> str:
    """Compile the pipeline to a KFP IR YAML spec locally (no cluster).

    Args:
        output_path: Where to write the compiled IR (``.yaml``).

    Returns:
        The path the IR spec was written to.
    """
    compiler.Compiler().compile(clinops_pipeline, package_path=output_path)
    return output_path


if __name__ == "__main__":
    # Local entry point: compile the pipeline to an IR spec. Submit to a cluster
    # separately (see README) — this never stands up kind or submits a run.
    compile_pipeline("clinops_pipeline.yaml")
