"""Prefect flow — the same ClinOps pipeline expressed as a Prefect flow.

This flow **only wraps** the orchestrator-agnostic core in
`src/clinops/tasks/pipeline_tasks.py`. Each Prefect ``@task`` is a thin wrapper
that calls one of the shared step functions in order:

    extract -> parse_fhir -> build_features -> train -> evaluate -> promote -> package

No pipeline logic lives here — change a step in `pipeline_tasks.py` and every
orchestrator (Airflow/Prefect/Kubeflow) inherits it (see CLAUDE.md). This is the
Prefect-idiomatic expression of the *same* linear graph the Airflow DAG encodes,
so the two stay honestly comparable.

The flow is **manual-run only** (``make prefect-run`` / ``python clinops_flow.py``);
no deployment or schedule is created here. Step outputs flow downstream as task
return values; ``Path`` returns are stringified to mirror the Airflow DAG's
XCom-safe marshaling, and the core functions accept those strings transparently.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from prefect import flow, task
from prefect.tasks import Task

from clinops.tasks import pipeline_tasks

FLOW_NAME = "clinops_pipeline"

# Ordered pipeline steps. Each value is the real single-source-of-truth callable
# in pipeline_tasks that the matching Prefect task wraps — nothing is re-implemented.
PIPELINE_STEPS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("extract", pipeline_tasks.extract),
    ("parse_fhir", pipeline_tasks.parse_fhir),
    ("build_features", pipeline_tasks.build_features),
    ("train", pipeline_tasks.train),
    ("evaluate", pipeline_tasks.evaluate),
    ("promote", pipeline_tasks.promote),
    ("package", pipeline_tasks.package),
)


def _marshal(value: Any) -> Any:
    """Make a step's return value serializable downstream (``Path`` -> ``str``)."""
    return str(value) if isinstance(value, Path) else value


def _make_task(step: Callable[..., Any]) -> Task:
    """Build the thin Prefect task for one core step — no pipeline logic here.

    The returned task pulls the single upstream value (if any), calls the matching
    ``pipeline_tasks`` function, and returns a serializable value. The wrapped core
    function is exposed via ``__wrapped_step__`` (on ``Task.fn``) so structural
    tests can assert real wiring, mirroring the Airflow DAG.

    Args:
        step: The ``pipeline_tasks`` function this task wraps.

    Returns:
        The Prefect ``Task`` for this step.
    """

    def runner(upstream: Any = None) -> Any:
        """Execute the wrapped pipeline step for this Prefect task."""
        if upstream is None:
            return _marshal(step())
        return _marshal(step(upstream))

    runner.__name__ = f"run_{step.__name__}"
    runner.__doc__ = f"Prefect task wrapping clinops.tasks.pipeline_tasks.{step.__name__}."
    # Expose the wrapped core callable so structural tests can assert real wiring.
    runner.__wrapped_step__ = step  # type: ignore[attr-defined]
    return task(runner, name=step.__name__)


# One Prefect task per step, in pipeline order — the structural analog of the
# Airflow DAG's operator list.
PREFECT_TASKS: tuple[tuple[str, Task], ...] = tuple(
    (task_id, _make_task(step)) for task_id, step in PIPELINE_STEPS
)


@flow(name=FLOW_NAME)
def clinops_flow() -> Any:
    """Run the ClinOps pipeline core as a linear Prefect flow.

    Calls each wrapped step in order, passing each task's return value to the next
    (extract -> ... -> package). Sequential data passing encodes the linear
    dependency graph the Airflow DAG declares explicitly.

    Returns:
        The final ``package`` step's serving manifest.
    """
    result: Any = None
    for _task_id, pipeline_task in PREFECT_TASKS:
        result = pipeline_task(result)
    return result


if __name__ == "__main__":
    # Local entry point used by `make prefect-run`. Runs the flow in-process.
    clinops_flow()
