# Orchestrator Trade-off Comparison

> Status: scaffold. **No results yet** — this document will be filled in *after*
> the same pipeline has actually been run on each orchestrator. No fabricated
> metrics (see CLAUDE.md). The tables below are the structure to fill, not claims.

## The setup

All three orchestrators run the **same** pipeline by wrapping the shared step
functions in [`src/clinops/tasks/pipeline_tasks.py`](../src/clinops/tasks/pipeline_tasks.py)
(`extract → parse_fhir → build_features → train → evaluate → promote → package`).
None re-implement the logic. This isolates the orchestrator as the only variable.

## What we compare

| Dimension | Airflow | Prefect | Kubeflow |
|-----------|---------|---------|----------|
| Authoring model | _TBD_ | _TBD_ | _TBD_ |
| Local setup effort | _TBD_ | _TBD_ | _TBD_ |
| Scheduling | _TBD_ | _TBD_ | _TBD_ |
| Retries / error handling | _TBD_ | _TBD_ | _TBD_ |
| Observability / UI | _TBD_ | _TBD_ | _TBD_ |
| Parameterization | _TBD_ | _TBD_ | _TBD_ |
| Packaging / deployment | _TBD_ | _TBD_ | _TBD_ |
| Infra footprint | _TBD_ | _TBD_ | _TBD_ |
| Best fit for... | _TBD_ | _TBD_ | _TBD_ |

## How each wraps the core

- **Airflow** ([`clinops_dag.py`](../orchestration/airflow/dags/clinops_dag.py)) —
  each task is an operator that calls one `pipeline_tasks.*` function; runs on a
  schedule via Docker Compose. _Notes: TBD._
- **Prefect** ([`clinops_flow.py`](../orchestration/prefect/clinops_flow.py)) —
  each `@task` wraps one step function; a `@flow` composes them. _Notes: TBD._
- **Kubeflow** ([`pipeline.py`](../orchestration/kubeflow/pipeline.py)) —
  each KFP component wraps one step function; compiled and run once on a local
  `kind` cluster. _Notes: TBD._

## Verdict

_To be written once all three have been run. Summarize when you'd reach for each._
