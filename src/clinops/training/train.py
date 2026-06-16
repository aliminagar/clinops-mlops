"""Train and evaluate the baseline models, with the MLflow tracking/registry spine.

Orchestrates the modelling stage end to end, with no leakage:

1. Load the feature store and drop zero-variance columns.
2. Build the fixed, stratified 60/20/20 split (reused across models).
3. Fit three candidates on the **train** fold only (no leakage): a class-balanced
   logistic regression, an imbalance-weighted LightGBM, and a PyTorch feed-forward
   challenger — all on identical features, so the comparison is apples-to-apples.
4. Score each model with stratified 5-fold PR-AUC on train (mean ± std), then on
   the validation fold (tuning) and finally the test fold (reported once). Three
   test operating points are reported side by side: the fixed 0.5 threshold, the
   max-F1 threshold, and the **deployed** F2 (recall-weighted) threshold. Tuned
   thresholds are chosen on **validation** and applied once to test (no leakage).
5. Persist fitted models to ``models/`` and a metrics JSON + calibration curve +
   SHAP summary (LightGBM) to ``reports/``.
6. Log every model to MLflow under one parent run (nested child runs) through a
   flavor-agnostic pyfunc wrapper, then **promote the champion** — the model with
   the highest CV PR-AUC — by registering it and pointing the ``champion`` alias
   at it. Any flavor can win, and the registered champion serves identically.

Imbalance is handled by class weighting only — no SMOTE/resampling. The headline
metric is PR-AUC; see :mod:`clinops.training.evaluate`.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from clinops.config import settings
from clinops.models.sklearn_models import (
    build_lightgbm,
    build_logistic_regression,
    scale_pos_weight_from,
)
from clinops.models.torch_model import TorchMLPClassifier
from clinops.registry import promote as registry_promote
from clinops.serving.proba_model import ProbaModel
from clinops.training import evaluate, split

logger = logging.getLogger(__name__)

# Continuous features have more than two distinct values; the rest are binary
# indicators that are passed through the logistic-regression scaler unchanged.
_BINARY_NUNIQUE = 2

# Decision threshold for precision/recall/F1 (PR-AUC, the headline, is
# threshold-free). Kept explicit so the reported point is unambiguous.
DEFAULT_THRESHOLD = 0.5

# The tree model that gets a SHAP summary (TreeExplainer only fits tree models).
_SHAP_MODEL = "lightgbm"
_MLFLOW_ARTIFACT_PATH = registry_promote.MODEL_ARTIFACT_PATH


def continuous_features(features: pd.DataFrame) -> list[str]:
    """Return the names of columns to standardize (more than two distinct values)."""
    return [col for col in features.columns if features[col].nunique() > _BINARY_NUNIQUE]


def _model_factories(
    features: pd.DataFrame,
    train_target: pd.Series,
    *,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Build per-model CV factories and final (train-fitted) estimators.

    Each model exposes a ``make`` factory that rebuilds a fresh estimator from a
    fold's train labels (so LightGBM's ``scale_pos_weight`` is recomputed per
    fold) and a ``final`` estimator parameterized from the full train fold.
    """
    cont = continuous_features(features)
    train_spw = scale_pos_weight_from(train_target)

    def make_logreg(_fold_target: pd.Series) -> Any:
        return build_logistic_regression(cont, seed=seed)

    def make_lgbm(fold_target: pd.Series) -> Any:
        return build_lightgbm(scale_pos_weight_from(fold_target), seed=seed)

    def make_torch(_fold_target: pd.Series) -> Any:
        # pos_weight is recomputed from the fold's own labels inside fit().
        return TorchMLPClassifier(seed=seed)

    return {
        "logistic_regression": {
            "make": make_logreg,
            "final": build_logistic_regression(cont, seed=seed),
            "scale_pos_weight": None,
            "class_weight": "balanced",
        },
        "lightgbm": {
            "make": make_lgbm,
            "final": build_lightgbm(train_spw, seed=seed),
            "scale_pos_weight": train_spw,
            "class_weight": "scale_pos_weight",
        },
        "torch_mlp": {
            "make": make_torch,
            "final": TorchMLPClassifier(seed=seed),
            "scale_pos_weight": train_spw,
            "class_weight": "pos_weight",
        },
    }


def _positive_proba(model: Any, features: pd.DataFrame) -> Any:
    """Return the positive-class probability column from ``predict_proba``."""
    return model.predict_proba(features)[:, 1]


def _train_and_eval(
    spec: dict[str, Any],
    data: dict[str, tuple[pd.DataFrame, pd.Series]],
    *,
    seed: int,
    k: int,
    threshold: float,
    thr_rule: str,
    thr_beta: float,
    precision_floor: float,
) -> tuple[Any, dict[str, Any], Any]:
    """Cross-validate, fit, and evaluate one model (mlflow-agnostic).

    Returns the fitted model, its metrics block, and the test-set positive-class
    probabilities (kept for the shared calibration plot).
    """
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]
    x_test, y_test = data["test"]

    cv = evaluate.cross_val_average_precision(spec["make"], x_train, y_train, k=k, seed=seed)

    model = spec["final"]
    model.fit(x_train, y_train)

    val_proba = _positive_proba(model, x_val)
    test_proba = _positive_proba(model, x_test)
    val_metrics = evaluate.compute_metrics(y_val, val_proba, threshold=threshold)
    test_metrics = evaluate.compute_metrics(y_test, test_proba, threshold=threshold)

    # Three side-by-side test operating points. The fixed 0.5 point and the
    # max-F1 point are surfaced for context; the deployed point is the configured
    # rule (default F2, recall-weighted). Tuned thresholds come from validation
    # only and are applied once to test (no leakage).
    fixed_point = {"source": "fixed", **evaluate.operating_point(y_test, test_proba, threshold)}
    max_f1_point = evaluate.select_and_apply_threshold(
        y_val, val_proba, y_test, test_proba, rule=evaluate.RULE_MAX_FBETA, beta=1.0
    )
    deployed_point = evaluate.select_and_apply_threshold(
        y_val,
        val_proba,
        y_test,
        test_proba,
        rule=thr_rule,
        beta=thr_beta,
        precision_floor=precision_floor,
    )
    deployed_point["deployed"] = True
    operating_points = {
        "fixed_0p5": fixed_point,
        "max_f1": max_f1_point,
        "f2": deployed_point,
    }

    block: dict[str, Any] = {
        "scale_pos_weight": spec["scale_pos_weight"],
        "class_weight": spec["class_weight"],
        "cv_train": cv,
        "val": val_metrics,
        "test": test_metrics,
        "operating_points": operating_points,
        "deployed_threshold": {
            "value": deployed_point["threshold"],
            "rule": deployed_point["rule"],
            "beta": deployed_point.get("beta"),
            "source": deployed_point["source"],
            "point": "f2",
        },
    }
    return model, block, test_proba


def _mlflow_params(
    name: str, block: dict[str, Any], dataset: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Flatten a model's configuration into MLflow params."""
    deployed = block["deployed_threshold"]
    params: dict[str, Any] = {
        "model_type": name,
        "seed": seed,
        "class_weight": block["class_weight"],
        "scale_pos_weight": block["scale_pos_weight"],
        "n_features": dataset["n_features"],
        "n_features_dropped": len(dataset["dropped_zero_variance"]),
        "features": ",".join(dataset["features"]),
        "dropped_features": ",".join(dataset["dropped_zero_variance"]) or "none",
        "cv_k": block["cv_train"]["k"],
        "operating_point": deployed["point"],
        "operating_rule": deployed["rule"],
        "operating_threshold_source": deployed["source"],
    }
    if deployed["beta"] is not None:
        params["operating_beta"] = deployed["beta"]
    return params


def _mlflow_metrics(block: dict[str, Any]) -> dict[str, float]:
    """Flatten a model's metrics into a single MLflow metrics mapping."""
    cv, val, test = block["cv_train"], block["val"], block["test"]
    metrics = {
        "cv_pr_auc_mean": cv["mean"],
        "cv_pr_auc_std": cv["std"],
        "operating_threshold": block["deployed_threshold"]["value"],
    }
    for split_name, values in (("val", val), ("test", test)):
        for key in ("pr_auc", "roc_auc", "precision", "recall", "f1", "brier"):
            metrics[f"{split_name}_{key}"] = values[key]
    # Each of the three test operating points: precision/recall/f1 + threshold.
    for point_name, point in block["operating_points"].items():
        for key in ("threshold", "precision", "recall", "f1"):
            metrics[f"test_{point_name}_{key}"] = point[key]
    return metrics


def _log_proba_model(mlflow: Any, estimator: Any, x_sample: pd.DataFrame) -> str:
    """Log a fitted estimator as a flavor-agnostic pyfunc model; return its URI.

    Every model is wrapped in :class:`ProbaModel` and logged with a signature
    inferred from the features, so ``mlflow.pyfunc`` serves positive-class
    probabilities regardless of the underlying framework.

    The signature is declared in ``float64`` for every column: the feature matrix
    mixes integer indicators and floats, and integer schema columns reject the
    float payloads the serving layer sends (MLflow's documented integer pitfall).
    """
    x_float = x_sample.astype("float64")
    proba = estimator.predict_proba(x_sample)[:, 1]
    signature = mlflow.models.infer_signature(x_float, proba)
    info = mlflow.pyfunc.log_model(
        name=_MLFLOW_ARTIFACT_PATH,
        python_model=ProbaModel(estimator),
        signature=signature,
        input_example=x_float.head(1),
    )
    return str(info.model_uri)


def run_training(
    feature_path: Path | None = None,
    *,
    splits_path: Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
    seed: int | None = None,
    k: int = 5,
    threshold: float = DEFAULT_THRESHOLD,
    run_shap: bool = True,
    track: bool = True,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    registered_model_name: str | None = None,
    threshold_rule: str = evaluate.RULE_MAX_FBETA,
    threshold_beta: float | None = None,
    precision_floor: float = 0.5,
) -> dict[str, Any]:
    """Train both baselines, evaluate them, log to MLflow, and persist artifacts.

    Args:
        feature_path: Feature-store parquet; defaults to the configured path.
        splits_path: Persisted split-label parquet; defaults to the configured path.
        models_dir: Output dir for fitted models; defaults to ``settings.models_dir``.
        reports_dir: Output dir for metrics/plots; defaults to ``settings.reports_dir``.
        seed: Random seed; defaults to ``settings.random_seed``.
        k: Number of cross-validation folds on the train set.
        threshold: Fixed decision threshold for the @0.5 operating point.
        run_shap: Whether to compute and save the LightGBM SHAP summary plot.
        track: Whether to log to MLflow (silently skipped if MLflow is absent).
        tracking_uri: MLflow tracking URI; defaults to ``settings.mlflow_tracking_uri``.
        experiment: MLflow experiment name; defaults to settings.
        registered_model_name: Registry name for the lead model; defaults to settings.
        threshold_rule: Operating-point rule (see :func:`evaluate.select_threshold`).
        threshold_beta: F-beta for the threshold rule; defaults to ``settings.threshold_beta``.
        precision_floor: Precision target for the precision-floor rule.

    Returns:
        The metrics summary (also written to ``reports/metrics.json``). The
        ``mlflow`` key holds the run IDs and registered model version, or ``None``
        when tracking is disabled.
    """
    resolved_seed = settings.random_seed if seed is None else seed
    resolved_beta = settings.threshold_beta if threshold_beta is None else threshold_beta
    out_models = Path(models_dir) if models_dir is not None else settings.models_dir
    out_reports = Path(reports_dir) if reports_dir is not None else settings.reports_dir
    out_models.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    # 1-2. Load features (zero-variance dropped) and build the fixed splits.
    source = (
        Path(feature_path)
        if feature_path is not None
        else settings.data_processed / "features.parquet"
    )
    frame = pd.read_parquet(source)
    if settings.target_name not in frame.columns:
        raise ValueError(f"Feature store {source!s} is missing target {settings.target_name!r}.")
    target_all = frame[settings.target_name].astype(int)
    features_raw = frame.drop(columns=[settings.target_name])
    features, dropped = split.drop_zero_variance(features_raw)

    splits = split.get_or_create_splits(target_all, path=splits_path, seed=resolved_seed)
    data = {name: split.split_xy(features, target_all, splits, name) for name in split.SPLIT_NAMES}

    factories = _model_factories(features, data["train"][1], seed=resolved_seed)
    dataset_info: dict[str, Any] = {
        "n_total": int(len(features)),
        "n_features": int(features.shape[1]),
        "features": list(features.columns),
        "dropped_zero_variance": dropped,
        "split_counts": {name: int((splits == name).sum()) for name in split.SPLIT_NAMES},
        "prevalence_overall": float(target_all.mean()),
    }

    # Optional MLflow setup (lazy import so the stage runs without MLflow).
    mlflow: Any = None
    if track:
        try:
            import mlflow as _mlflow
            import mlflow.lightgbm  # noqa: F401
            import mlflow.sklearn  # noqa: F401

            mlflow = _mlflow
        except ImportError:
            logger.warning("MLflow not installed; continuing without tracking.")
            track = False

    resolved_uri = tracking_uri or settings.mlflow_tracking_uri
    resolved_experiment = experiment or settings.mlflow_experiment_name
    reg_name = registered_model_name or settings.registered_model_name
    if track:
        # MLflow 3.x gates the local file store behind an opt-in; the registry is
        # supported on it, which keeps the simple ``mlruns/`` default working.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(resolved_uri)
        mlflow.set_experiment(resolved_experiment)

    model_metrics: dict[str, Any] = {}
    test_calibration: dict[str, tuple[Any, Any]] = {}
    registered: dict[str, Any] | None = None
    parent_run_id: str | None = None

    parent_cm: Any = mlflow.start_run(run_name="baselines-comparison") if track else nullcontext()
    with parent_cm as parent:
        if track:
            parent_run_id = parent.info.run_id

        # 3-5. Per model: cross-validate on train, fit, evaluate, log, persist.
        for name, spec in factories.items():
            child_cm: Any = mlflow.start_run(run_name=name, nested=True) if track else nullcontext()
            with child_cm as child:
                model, block, test_proba = _train_and_eval(
                    spec,
                    data,
                    seed=resolved_seed,
                    k=k,
                    threshold=threshold,
                    thr_rule=threshold_rule,
                    thr_beta=resolved_beta,
                    precision_floor=precision_floor,
                )
                test_calibration[name] = evaluate.calibration_points(data["test"][1], test_proba)

                model_path = out_models / f"{name}.joblib"
                joblib.dump(model, model_path)
                block["model_path"] = str(model_path)

                shap_path: Path | None = None
                if name == _SHAP_MODEL and run_shap:
                    shap_path = evaluate.save_shap_summary(
                        model,
                        data["test"][0],
                        out_reports / "shap_summary_lightgbm.png",
                        seed=resolved_seed,
                    )

                if track:
                    run_id = child.info.run_id
                    block["mlflow_run_id"] = run_id
                    mlflow.set_tag("operating_point", block["deployed_threshold"]["point"])
                    mlflow.log_params(_mlflow_params(name, block, dataset_info, resolved_seed))
                    mlflow.log_metrics(_mlflow_metrics(block))
                    # Every flavor is logged through the same pyfunc proba wrapper
                    # so any promoted champion serves identically; promotion later
                    # re-resolves these logged models from the child runs.
                    _log_proba_model(mlflow, model, data["train"][0])
                    if shap_path is not None:
                        mlflow.log_artifact(str(shap_path))

                model_metrics[name] = block
                f1_pt = block["operating_points"]["max_f1"]
                f2_pt = block["operating_points"]["f2"]
                logger.info(
                    "%s — CV PR-AUC=%.4f±%.4f | test PR-AUC=%.4f | "
                    "max-F1@%.3f R=%.3f | F2@%.3f R=%.3f (deployed)",
                    name,
                    block["cv_train"]["mean"],
                    block["cv_train"]["std"],
                    block["test"]["pr_auc"],
                    f1_pt["threshold"],
                    f1_pt["recall"],
                    f2_pt["threshold"],
                    f2_pt["recall"],
                )

        # Shared calibration curve (test fold) for all models.
        calibration_path = evaluate.plot_calibration(
            test_calibration, out_reports / "calibration_test.png"
        )

        # Champion/challenger promotion. Selection (highest CV PR-AUC, apples-to-
        # apples) lives once in registry.promote — used here in-memory for the
        # summary, and re-resolved from MLflow by promote_champion when tracking.
        cv_scores = {name: model_metrics[name]["cv_train"]["mean"] for name in model_metrics}
        champion_name = registry_promote.select_champion(cv_scores)
        champion = {
            "model": champion_name,
            "cv_pr_auc_mean": model_metrics[champion_name]["cv_train"]["mean"],
            "test_pr_auc": model_metrics[champion_name]["test"]["pr_auc"],
        }
        logger.info("Champion by CV PR-AUC: %s (%.4f)", champion_name, champion["cv_pr_auc_mean"])

        mlflow_block: dict[str, Any] | None = None
        if track:
            assert parent_run_id is not None  # set above whenever track is on
            promotion = registry_promote.promote_champion(
                parent_run_id, reg_name, tracking_uri=resolved_uri
            )
            registered = {
                "name": reg_name,
                "version": promotion.version,
                "alias": registry_promote.CHAMPION_ALIAS,
                "champion_model": promotion.winner,
            }
            mlflow_block = {
                "tracking_uri": resolved_uri,
                "experiment": resolved_experiment,
                "parent_run_id": parent_run_id,
                "runs": {name: model_metrics[name]["mlflow_run_id"] for name in model_metrics},
                "registered_model": registered,
            }

        summary: dict[str, Any] = {
            "seed": resolved_seed,
            "headline_metric": evaluate.HEADLINE_METRIC,
            "decision_threshold": threshold,
            "dataset": dataset_info,
            "champion": champion,
            "models": model_metrics,
            "artifacts": {
                "calibration_plot": str(calibration_path),
                "shap_summary": (
                    str(out_reports / "shap_summary_lightgbm.png") if run_shap else None
                ),
            },
            "mlflow": mlflow_block,
        }

        metrics_path = out_reports / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Wrote metrics to %s", metrics_path)

        if track:
            mlflow.log_artifact(str(calibration_path))
            mlflow.log_artifact(str(metrics_path))

    return summary


def main() -> None:
    """CLI entry point: run training against the configured feature store."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_training()


if __name__ == "__main__":
    main()
