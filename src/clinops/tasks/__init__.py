"""Tasks package: the orchestrator-agnostic pipeline core.

Every orchestrator (Airflow, Prefect, Kubeflow) imports and wraps the functions
in `pipeline_tasks` — none of them re-implement pipeline logic.
"""
