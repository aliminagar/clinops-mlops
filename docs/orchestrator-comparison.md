# Orchestrator trade-off: Airflow vs. Prefect vs. Kubeflow

This is the headline deliverable of ClinOps: a three-way comparison of **Apache
Airflow**, **Prefect**, and **Kubeflow Pipelines (KFP)** running the *same* ML
pipeline. It is a judgment doc — it contains reasoned engineering opinions, clearly
labelled as such, not measured benchmarks.

## The thesis: one pipeline, three wrappers

The pipeline logic lives **once**, as plain Python functions in
[`src/clinops/tasks/pipeline_tasks.py`](../src/clinops/tasks/pipeline_tasks.py):

```
extract → parse_fhir → build_features → train → evaluate → promote → package
```

Every orchestrator **imports and wraps** those seven functions; **none
re-implements them** (the non-negotiable "orchestrator-agnostic rule" in
[CLAUDE.md](../CLAUDE.md)). The Airflow DAG, the Prefect flow, and the KFP pipeline
all reference the identical `pipeline_tasks.extract`, `pipeline_tasks.train`, etc.
— each one's test even asserts the wrapper points at the real core callable (a
`__wrapped_step__` handle) rather than a local copy.

**This is what makes the comparison honest.** Because the work done in each step is
byte-for-byte the same code, every difference you see below is *purely
orchestration* — execution model, data passing, infrastructure, ergonomics — and
never a difference in what the pipeline computes. There is no "the Airflow version
trains a different model" confound. There is exactly one model pipeline, observed
through three lenses.

## The shared core all three wrap

Each step is a thin function that delegates to a `clinops.*` module and returns a
small, serializable handle to the next step:

| Step | Returns | Consumed by |
|---|---|---|
| `extract` | `Path` to `data/raw/` bundles | `parse_fhir` |
| `parse_fhir` | `Path` to `data/interim/cohort.parquet` | `build_features` |
| `build_features` | `Path` to the processed feature store | `train` |
| `train` | `dict` of MLflow run references | `evaluate` |
| `evaluate` | `dict` of per-model test metrics | `promote` |
| `promote` | `dict` (champion model, version, F2 threshold) | `package` |
| `package` | `dict` serving manifest | — |

Two properties of the core shape every wrapper. First, the steps pass **paths and
small dicts**, not large in-memory frames — heavy state (the cohort, the feature
store, the MLflow registry) lives on disk / in the tracking server, and steps hand
each other *references*. Second, the path-returning steps accept either a `Path`
or a `str`, so an orchestrator that can only marshal strings between steps still
works transparently. Both properties are what let three very different execution
models wrap the same code without friction.

## How each orchestrator wraps the core

### Airflow — XCom between operators, on a Compose stack

[`orchestration/airflow/dags/clinops_dag.py`](../orchestration/airflow/dags/clinops_dag.py)
builds one `PythonOperator` per step from a `PIPELINE_STEPS` tuple and chains them
linearly with `set_downstream`. Each operator's callable pulls its single upstream
value from **XCom** (`context["ti"].xcom_pull(task_ids=...)`), calls the core
function, and pushes the result back. Because XCom values must be
JSON-serializable, a small `_xcom_safe` helper stringifies `Path` returns — the
core accepts those strings, so nothing else changes.

- **Trigger / run:** manual only (`schedule=None`, `catchup=False`, `retries=0`);
  fire with `airflow dags trigger clinops_pipeline`.
- **Infra to stand up:** a multi-container **docker-compose** stack
  ([`docker-compose.airflow.yml`](../orchestration/airflow/docker-compose.airflow.yml)):
  Postgres metadata DB, a one-shot `airflow-init`, a webserver (UI on `:8080`), and
  a scheduler, on `apache/airflow:2.9.0` with `LocalExecutor`. The `clinops`
  package is bind-mounted to `/opt/airflow/src` and the ML deps are installed at
  container start via `_PIP_ADDITIONAL_REQUIREMENTS`; MLflow is reached at
  `host.docker.internal:5000`.

### Prefect — in-process returns, zero standing infra

[`orchestration/prefect/clinops_flow.py`](../orchestration/prefect/clinops_flow.py)
wraps each step as an `@task` (built from the same `PIPELINE_STEPS` tuple) and a
single `@flow` (`clinops_flow`) chains them by **passing each task's return value
directly to the next** — ordinary Python data flow, in one process. A `_marshal`
helper stringifies `Path` returns to mirror Airflow's behaviour and keep the two
honestly comparable, but Prefect could pass the `Path` object itself.

- **Trigger / run:** `python orchestration/prefect/clinops_flow.py` (or
  `make prefect-run`) runs the flow **in-process** — no server, no deployment, no
  agent required for a local run.
- **Infra to stand up:** none for local execution (just the project venv). A
  Prefect server / Cloud workspace is optional and only needed for scheduling,
  a remote UI, and durable run history.

### Kubeflow — containerized steps, artifacts across boundaries

[`orchestration/kubeflow/pipeline.py`](../orchestration/kubeflow/pipeline.py) is
the structurally different one. Each step is a `@dsl.component` — a **hermetic**
function that `import`s `pipeline_tasks` *inside the function body* because it is
extracted, shipped, and executed as its **own container** on Kubernetes. Steps
cannot hand each other Python objects in memory; they exchange **KFP parameters /
file artifacts** across container boundaries. The wrappers return `str` (for the
path steps) and `dict` (for the later steps), and the `@dsl.pipeline`
(`clinops_pipeline`) wires `task.output → next component`, which KFP turns into the
parameter/artifact dependency graph.

- **Trigger / run:** `python orchestration/kubeflow/pipeline.py` **compiles** the
  pipeline to a KFP IR YAML locally (no cluster); the compiled spec is then
  uploaded/submitted to a cluster separately.
- **Infra to stand up:** a Kubernetes cluster. Locally that means **kind**
  (Kubernetes-in-Docker) via
  [`kind-config.yaml`](../orchestration/kubeflow/kind-config.yaml) (which exposes
  the ml-pipeline UI on host port `8090`), plus a Kubeflow Pipelines install, plus
  a **container image that has `clinops` and its ML deps baked in** (the default
  `python:3.11` base image does not). This is materially more setup than the other
  two.

## Comparison table

| Axis | Airflow | Prefect | Kubeflow (KFP) |
|---|---|---|---|
| **Execution model** | In-process per task (LocalExecutor), one host | In-process, single Python process | One **container per step** on Kubernetes |
| **Data passing between steps** | XCom (JSON-serializable; `Path`→`str`) | Direct Python return values, in-memory | Parameters / file **artifacts** across container boundaries |
| **Trigger** | `dags trigger` / schedule (here: manual) | `python` call / `make prefect-run` (here: in-process) | Compile to IR → submit to cluster |
| **Infra to operate** | docker-compose: Postgres + webserver + scheduler | **None** for local run (venv only) | k8s cluster (kind) + KFP install + custom image |
| **Local dev friction** | Medium — bring up the stack, mount code, install deps in-container | **Lowest** — run the file | **Highest** — cluster + image build before anything runs |
| **Scaling model** | Scheduler + executors (Celery/K8s executor for scale-out) | Work pools / workers; Dask/Ray task runners | Native k8s: per-step pods, resource limits, parallelism, autoscaling |
| **Step isolation** | Shared host/env per worker | Shared process | **Strong** — each step its own container/image/resources |
| **Observability / UI** | Mature DAG UI, logs, run history (`:8080`) | Modern flow-run UI (server/Cloud); rich local logs | KFP UI: DAG, artifacts, **experiment lineage** (`:8090`) |
| **Retries & scheduling** | First-class, battle-tested cron-style scheduling | First-class, Pythonic retries/schedules | Per-step retries/caching; scheduling via recurring runs |
| **ML-native features** | General-purpose; ML is convention | General-purpose; Pythonic | **ML-first**: artifact lineage, metadata, experiment tracking |
| **Learning curve** | Steeper (DAG concepts, operators, deployment) | **Gentlest** (it's just decorated Python) | Steepest (k8s + KFP SDK + containerization) |
| **Best-fit use case** | Scheduled production batch, mixed (not just ML) workloads | Fast local/iterative dev; Python-centric teams | Multi-team, k8s-native production ML at scale |

## The real trade-off

Stripped of detail, the axis is **lightweight Python-native orchestration
(Prefect, then Airflow) vs. heavyweight Kubernetes-native ML orchestration
(Kubeflow)**.

**What the lightweight end buys you.** Prefect's whole model is "your pipeline is
just Python functions, and the orchestrator is a decorator." For ClinOps that is an
almost perfect fit: the core *already is* plain functions returning small handles,
so the wrapper is nearly transparent and a full run is one command with zero
standing infrastructure. Airflow sits a step up the ladder — you pay for a
Compose stack and the XCom serialization dance, but you get a mature scheduler,
durable run history, retries, and a UI that operations teams already know. Both run
the entire pipeline on a single box.

**What Kubeflow buys you, and what it costs.** Kubeflow's container-per-step model
is a genuine architectural advantage, not ceremony: each step gets its own image,
its own resource requests/limits, and its own failure domain. The `train` step can
demand a GPU node while `parse_fhir` runs on a cheap CPU pod; steps can fan out and
autoscale; and KFP records artifact lineage and experiment metadata as first-class
objects. That isolation and elasticity is exactly what you want when many teams
ship many pipelines onto shared infrastructure.

The cost is equally real, and this repo makes it concrete. To run the KFP pipeline
for real you must stand up a Kubernetes cluster (kind locally), install Kubeflow
Pipelines, **and build a container image that contains `clinops` and its ML deps**
— and the data no longer flows in memory but is marshalled across container
boundaries as artifacts. For ClinOps as it exists — one synthetic Synthea cohort
(5,626 patients; see [metrics.md](metrics.md)) trained on a single machine — that
operational cost is **not justified**: the container isolation and autoscaling
solve problems this workload does not have. *(Reasoned engineering judgment, not a
measured result.)* The same calculus inverts for production clinical ML across
multiple teams, with heterogeneous hardware needs, strict per-step isolation/audit
requirements, and an existing k8s platform — there, Kubeflow's costs are mostly
*already paid* and its advantages become the point.

## When I'd choose each

*(Engineering judgment, framed as decision guidance.)*

- **Choose Prefect** when the team is Python-first, the priority is fast local
  iteration, and you don't already operate scheduling infrastructure. Best for
  small teams, research/prototyping, and exactly the local-dev loop this repo lives
  in. Lowest friction to first successful run.
- **Choose Airflow** when you need a **scheduled, durable, production batch**
  orchestrator and want the boring, well-understood default — especially if the
  workload is broader than ML (ETL, reporting, mixed DAGs) and ops already knows
  Airflow. The Compose stack is a fair price for the maturity.
- **Choose Kubeflow** when you are **already on Kubernetes**, run ML across
  multiple teams, need per-step hardware (GPUs), strong isolation, and
  artifact/experiment lineage as a platform capability — i.e. production clinical
  ML at organizational scale, not a single-box pipeline.

The deciding questions, in order: *Do you already run Kubernetes? How many teams
and pipelines share the infra? Do steps need different hardware or hard isolation?*
If the answers are no / one / no, stay lightweight.

## For this project specifically

ClinOps is a single-machine pipeline on a synthetic cohort, optimized for
reproducibility and clarity. So:

- **Pragmatic default for local development: Prefect.** It matches the core's
  plain-function shape almost exactly and runs the whole pipeline with one command
  and no infrastructure — the least friction for "clone and run."
- **Production-style reference: Airflow.** It is the familiar scheduled-batch
  default; the bundled Compose stack is the closest thing here to a real deployment,
  with a scheduler and UI an operations team would recognize.
- **Scale-out path: Kubeflow.** It is the deliberate "what if this had to scale
  across a k8s platform" answer — implemented and compiled to prove portability of
  the same core, kept as the future direction rather than the day-one tool.

In short: develop on Prefect, deploy the reference on Airflow, and reach for
Kubeflow only when the org and the scale actually demand Kubernetes.

## Implementation status & honesty

- All three orchestrators are **implemented and structurally tested** — each test
  asserts the wrapper imports cleanly, exposes the seven steps in the correct
  order, and wires each step to the real `pipeline_tasks` callable (KFP
  additionally **compiles** to a valid IR spec).
- They are **not yet validated on live runtimes on this machine**: the Airflow
  Compose stack, a Prefect server run, and a kind/Kubeflow cluster have not been
  stood up here. The structural tests `importorskip` their orchestrator
  (`airflow` / `prefect` / `kfp` are not installed in the local venv), so they
  **skip locally and run where those deps are installed** (CI / the respective
  containers and clusters). The roadmap marks all three **🟡 Written**, not Done,
  for exactly this reason.
- **No runtime numbers are claimed.** This document contains **no measured latency,
  throughput, or benchmark figures** for any orchestrator — none were measured. The
  only quantitative results in the project are the model metrics in
  [metrics.md](metrics.md), which come from a real training run on the
  **synthetic** Synthea cohort (no real patients, no PHI).
- Opinions above (best-fit, "when I'd choose", the not-justified-here verdict on
  Kubeflow) are **reasoned engineering judgment**, not experimental findings.
