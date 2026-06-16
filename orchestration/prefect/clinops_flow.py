"""Prefect flow — the same ClinOps pipeline re-implemented as a Prefect flow.

This flow ONLY wraps the orchestrator-agnostic core in
`src/clinops/tasks/pipeline_tasks.py`. Each Prefect `@task` calls one of the
shared step functions; the `@flow` wires them together. Do NOT re-implement
pipeline logic here — see CLAUDE.md and `docs/orchestrator-comparison.md`.
"""

from __future__ import annotations

# from clinops.tasks import pipeline_tasks
# Each Prefect task should be a thin wrapper around a pipeline_tasks.* function.


def clinops_flow() -> object:
    """Define and return the ClinOps Prefect flow.

    Wraps `pipeline_tasks` step functions as Prefect tasks and composes them
    into a flow (extract -> ... -> package).
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Entry point used by `make prefect-run`.
    raise NotImplementedError
