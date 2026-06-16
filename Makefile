# ClinOps developer commands.
# NOTE: downstream pipeline steps (train/serve/etc.) are still stubs and will
# raise NotImplementedError until implemented. The ETL stage is real.

# --- Synthea (synthetic data generation) ---
SYNTHEA_JAR ?= synthea-with-dependencies.jar
SYNTHEA_URL ?= https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar
SYNTHEA_POP ?= 1000

.PHONY: setup synthea etl train serve test lint airflow-up prefect-run

setup:  ## Install dependencies and the package in editable mode
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

synthea:  ## Generate ~1000 synthetic patients as FHIR R4 into data/raw/ (needs Java 11+)
	@command -v java >/dev/null 2>&1 || { echo "ERROR: Java 11+ is required (see https://adoptium.net)"; exit 1; }
	@test -f $(SYNTHEA_JAR) || { echo "Downloading Synthea jar..."; curl -L -o $(SYNTHEA_JAR) "$(SYNTHEA_URL)"; }
	# FHIR R4 export enabled; output lands in data/raw/fhir/ (synthetic only, no PHI).
	# Synthea also emits hospital/practitioner bundles, which the loader skips.
	java -jar $(SYNTHEA_JAR) -p $(SYNTHEA_POP) \
		--exporter.fhir.export=true \
		--exporter.baseDirectory=data/raw

etl:  ## Run the ETL: load Synthea FHIR bundles -> parse -> features
	python -m clinops.tasks.pipeline_tasks --step etl

train:  ## Train sklearn baselines + PyTorch challenger, log to MLflow
	python -m clinops.tasks.pipeline_tasks --step train

serve:  ## Serve the champion model via BentoML /predict
	bentoml serve clinops.serving.bentoml_service:svc --reload

test:  ## Run the test suite
	pytest

lint:  ## Lint and format-check with ruff
	ruff check src tests
	ruff format --check src tests

airflow-up:  ## Bring up the Airflow stack (Docker Compose)
	docker compose -f orchestration/airflow/docker-compose.airflow.yml up -d

prefect-run:  ## Run the pipeline as a Prefect flow
	python orchestration/prefect/clinops_flow.py
