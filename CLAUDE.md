# CLAUDE.md — Working agreement for ClinOps

This file is the working agreement between the user and Claude Code on this repo.
Read it before making changes.

## Project goal

ClinOps is a portfolio MLOps project: **one orchestrator-agnostic ML pipeline**,
trained on synthetic FHIR R4 data (Synthea), executed on three orchestrators
(Airflow, Prefect, Kubeflow) as a documented trade-off comparison. MLflow is the
spine (tracking + registry); BentoML serves the registry's champion model at a
`/predict` REST endpoint. scikit-learn baselines and a PyTorch challenger compete
on the same features; the registry picks the winner.

## The orchestrator-agnostic rule (non-negotiable)

- The pipeline steps live **once** as plain Python functions in
  [`src/clinops/tasks/pipeline_tasks.py`](src/clinops/tasks/pipeline_tasks.py):
  `extract → parse_fhir → build_features → train → evaluate → promote → package`.
- Every orchestrator file (Airflow DAG, Prefect flow, Kubeflow pipeline) must
  **import and wrap** these functions. **Never re-implement pipeline logic** in an
  orchestrator file. If you need to change a step, change it in `pipeline_tasks.py`.

## Conventions

- **Stubs stay stubs until implemented.** A stub has a docstring stating its
  purpose plus `raise NotImplementedError` (or `pass`). Do not make something
  *look* finished that isn't. Implement intentionally, one component at a time.
- **No fabricated metrics in the README** (or anywhere). Do not invent accuracy,
  AUC, latency, or benchmark numbers. Report only real, reproduced results.
- **No real patient data, ever.** Synthetic Synthea data only; no PHI. The
  `data/` tree is git-ignored.
- **Python style:** `ruff` for lint + format, full **type hints** on public
  functions, docstrings on modules and public callables. Target Python 3.11.

## Build order

Implement in this sequence (each builds on the previous):

1. **ETL** — `etl/synthea_loader.py`, `etl/fhir_parser.py`
2. **Models** — `models/sklearn_models.py`, `models/torch_model.py`,
   `features/build_features.py`, `training/train.py`, `training/evaluate.py`
3. **MLflow** — wire tracking + registry into training and `registry/promote.py`
4. **BentoML** — `serving/bentoml_service.py` serves the champion
5. **Airflow** — `orchestration/airflow/dags/clinops_dag.py` wraps the core
6. **CI** — `.github/workflows/ci.yml` (lint + pytest) green
7. **Prefect** — `orchestration/prefect/clinops_flow.py` wraps the core
8. **Kubeflow** — `orchestration/kubeflow/pipeline.py` on a local `kind` cluster

Keep `pipeline_tasks.py` as the single source of truth throughout.

## Current status (checkpoint)
Built + tested: ETL, features, 3 models (logreg/lightgbm/torch_mlp), MLflow
tracking + registry + champion/challenger promotion, F2 threshold selection,
BentoML /predict serving (flavor-agnostic pyfunc). All docs synced to code+results.
Core pipeline now runs END-TO-END: ZERO NotImplementedError in the 7-step DAG path
(extract->...->package); promotion centralized in registry/promote.py (select on
CV PR-AUC, register, champion alias; idempotent).
Airflow DAG written + structurally tested (DagBag; importorskip on Windows) — 🟡
unrun on a live cluster.
Champion: logistic_regression via `champion` alias (currently v4, increments per
promotion), F2 threshold 0.813 (promoted on CV PR-AUC).
Cohort: Synthea -p 5000 -s 42 (5626 patients, ~3.5% readmission).
Gates: ruff/mypy/pytest all clean (59 passed, 1 skipped).

## Next up
1. Prefect  2. Kubeflow  3. orchestrator-comparison.md  4. GitHub push
   (CI verifies itself on push).
Minor optional: wire run_pipeline()/CLI (convenience entry points, not in the DAG path).
