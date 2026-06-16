"""Kubeflow Pipelines (KFP) definition for ClinOps.

Authored with the KFP SDK and run once on a local `kind` cluster to demonstrate
Kubernetes-native orchestration. Each KFP component ONLY wraps the
orchestrator-agnostic core in `src/clinops/tasks/pipeline_tasks.py` — it does
not re-implement pipeline logic. See CLAUDE.md and the comparison doc.
"""

from __future__ import annotations

# from clinops.tasks import pipeline_tasks
# Each KFP component should call one pipeline_tasks.* function and nothing more.


def clinops_pipeline() -> object:
    """Define and return the ClinOps KFP pipeline.

    Builds KFP components that wrap `pipeline_tasks` step functions and wires
    them into a pipeline (extract -> ... -> package) for compilation.
    """
    raise NotImplementedError
