"""Imbalance-aware evaluation: metrics, cross-validation, calibration, SHAP.

The headline metric is **PR-AUC (average precision)**, the appropriate summary
for a rare positive class (~3.5% readmission). Accuracy is deliberately *not*
reported, since a trivial all-negative classifier would score ~96% on it.
Alongside PR-AUC we report ROC-AUC, precision, recall, F1 (at a fixed decision
threshold), and the Brier score, plus a calibration curve and SHAP attributions.

Cross-validation (:func:`cross_val_average_precision`) is a stratified k-fold on
the **train** set only; the estimator for each fold is built fresh via a factory
so any class weighting is recomputed per fold (no cross-fold leakage).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# Headline metric key (PR-AUC / average precision).
HEADLINE_METRIC = "pr_auc"

# Operating-point selection rules for :func:`select_threshold`.
RULE_MAX_FBETA = "max_fbeta"
RULE_MIN_PRECISION = "min_precision"

# Matplotlib must render headless (no display) in CI / pipeline runs.
plt.switch_backend("Agg")


def compute_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute imbalance-aware classification metrics from scores.

    Args:
        y_true: True binary labels.
        y_score: Predicted positive-class probabilities.
        threshold: Decision threshold for precision/recall/F1.

    Returns:
        A metric-name -> value mapping. ``pr_auc`` (average precision) is the
        headline; also includes ``roc_auc``, ``precision``, ``recall``, ``f1``,
        ``brier``, plus ``threshold``, ``n``, ``n_positive`` and ``prevalence``.
        Accuracy is intentionally omitted.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred = (y_score_arr >= threshold).astype(int)

    n = int(y_true_arr.size)
    n_positive = int(y_true_arr.sum())
    return {
        "pr_auc": float(average_precision_score(y_true_arr, y_score_arr)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true_arr, y_score_arr)),
        "threshold": float(threshold),
        "n": float(n),
        "n_positive": float(n_positive),
        "prevalence": float(n_positive / n) if n else 0.0,
    }


def operating_point(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float,
) -> dict[str, float]:
    """Return precision/recall/F1 at a fixed decision threshold.

    Args:
        y_true: True binary labels.
        y_score: Predicted positive-class probabilities.
        threshold: Decision threshold.

    Returns:
        A mapping with ``threshold``, ``precision``, ``recall`` and ``f1``.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
    }


def select_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    rule: str = RULE_MAX_FBETA,
    beta: float = 1.0,
    precision_floor: float = 0.5,
) -> dict[str, Any]:
    """Choose a decision threshold from a precision-recall curve.

    Intended to be run on the **validation** scores only; the returned threshold
    is then applied once to the test set (no leakage). Two rules are supported:

    - ``"max_fbeta"`` — the threshold maximizing F-beta (``beta=1`` gives F1).
    - ``"min_precision"`` — the highest-recall threshold whose precision meets
      ``precision_floor``. If no threshold reaches the floor, falls back to the
      highest-precision threshold and flags ``met_precision_floor=False``.

    Args:
        y_true: True binary labels (validation).
        y_score: Predicted positive-class probabilities (validation).
        rule: Selection rule, one of :data:`RULE_MAX_FBETA` / :data:`RULE_MIN_PRECISION`.
        beta: Beta for the F-beta rule.
        precision_floor: Minimum precision target for the precision-floor rule.

    Returns:
        A mapping describing the chosen point: ``threshold``, ``rule``,
        ``source`` (``"validation"``), the val ``precision``/``recall``/``fbeta``
        there, plus ``beta`` or ``precision_floor`` and ``met_precision_floor``.

    Raises:
        ValueError: If ``rule`` is not recognized.
    """
    if rule not in (RULE_MAX_FBETA, RULE_MIN_PRECISION):
        raise ValueError(f"Unknown threshold rule {rule!r}.")

    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_score_arr)
    # precision_recall_curve appends an endpoint with no threshold; drop it so
    # precision/recall align with thresholds element-wise.
    precision, recall = precision[:-1], recall[:-1]

    if thresholds.size == 0:  # degenerate (single score) — keep the default 0.5
        chosen = 0.5
        index = 0
        met_floor = True
    elif rule == RULE_MAX_FBETA:
        beta2 = beta * beta
        denominator = beta2 * precision + recall
        numerator = (1.0 + beta2) * precision * recall
        fbeta = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > 0,
        )
        index = int(np.argmax(fbeta))
        chosen = float(thresholds[index])
        met_floor = True
    else:  # RULE_MIN_PRECISION
        eligible = np.flatnonzero(precision >= precision_floor)
        met_floor = bool(eligible.size)
        if met_floor:
            index = int(eligible[np.argmax(recall[eligible])])
        else:
            logger.warning(
                "No threshold reaches precision floor %.3f; falling back to max precision.",
                precision_floor,
            )
            index = int(np.argmax(precision))
        chosen = float(thresholds[index])

    chosen_precision = float(precision[index])
    chosen_recall = float(recall[index])
    denom = beta * beta * chosen_precision + chosen_recall
    chosen_fbeta = (
        (1.0 + beta * beta) * chosen_precision * chosen_recall / denom if denom > 0 else 0.0
    )

    result: dict[str, Any] = {
        "threshold": chosen,
        "rule": rule,
        "source": "validation",
        "precision": chosen_precision,
        "recall": chosen_recall,
        "fbeta": float(chosen_fbeta),
        "met_precision_floor": met_floor,
    }
    if rule == RULE_MAX_FBETA:
        result["beta"] = float(beta)
    else:
        result["precision_floor"] = float(precision_floor)
    return result


def select_and_apply_threshold(
    y_val: ArrayLike,
    val_score: ArrayLike,
    y_test: ArrayLike,
    test_score: ArrayLike,
    *,
    rule: str = RULE_MAX_FBETA,
    beta: float = 1.0,
    precision_floor: float = 0.5,
) -> dict[str, Any]:
    """Select a threshold on validation, then report the test operating point.

    Combines :func:`select_threshold` (validation only) with
    :func:`operating_point` (applied once to test), so the returned record holds
    both where the threshold came from and how it performs on the held-out test
    set. No leakage: the test labels never influence the threshold.

    Args:
        y_val: Validation labels.
        val_score: Validation positive-class probabilities.
        y_test: Test labels.
        test_score: Test positive-class probabilities.
        rule: Selection rule (see :func:`select_threshold`).
        beta: Beta for the F-beta rule.
        precision_floor: Precision target for the precision-floor rule.

    Returns:
        A mapping with ``threshold``, ``source`` (``"validation"``), ``rule``,
        the val ``val_precision``/``val_recall``, and the **test**
        ``precision``/``recall``/``f1`` at that threshold (plus ``beta`` or
        ``precision_floor`` depending on the rule).
    """
    selection = select_threshold(
        y_val, val_score, rule=rule, beta=beta, precision_floor=precision_floor
    )
    point = operating_point(y_test, test_score, selection["threshold"])
    record: dict[str, Any] = {
        "threshold": selection["threshold"],
        "source": selection["source"],
        "rule": selection["rule"],
        "val_precision": selection["precision"],
        "val_recall": selection["recall"],
        "precision": point["precision"],
        "recall": point["recall"],
        "f1": point["f1"],
    }
    if "beta" in selection:
        record["beta"] = selection["beta"]
    if "precision_floor" in selection:
        record["precision_floor"] = selection["precision_floor"]
        record["met_precision_floor"] = selection["met_precision_floor"]
    return record


def cross_val_average_precision(
    make_estimator: Callable[[pd.Series], Any],
    features: pd.DataFrame,
    target: pd.Series,
    *,
    k: int = 5,
    seed: int,
) -> dict[str, Any]:
    """Stratified k-fold PR-AUC on the train set, reported as mean ± std.

    A fresh estimator is built per fold via ``make_estimator(fold_train_target)``
    so class weighting is recomputed from each fold's train labels only.

    Args:
        make_estimator: Factory returning an unfitted estimator with
            ``fit``/``predict_proba``; receives the fold's train labels.
        features: Train-set feature matrix.
        target: Train-set target vector.
        k: Number of folds.
        seed: Seed for the (shuffled) stratified splitter.

    Returns:
        A mapping with ``metric`` (``"pr_auc"``), ``k``, ``mean``, ``std`` and the
        per-fold ``folds`` scores.
    """
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_idx, val_idx in splitter.split(features, target):
        x_tr, x_va = features.iloc[train_idx], features.iloc[val_idx]
        y_tr, y_va = target.iloc[train_idx], target.iloc[val_idx]
        estimator = make_estimator(y_tr)
        estimator.fit(x_tr, y_tr)
        proba = estimator.predict_proba(x_va)[:, 1]
        scores.append(float(average_precision_score(y_va, proba)))

    score_arr = np.asarray(scores, dtype=float)
    return {
        "metric": HEADLINE_METRIC,
        "k": k,
        "mean": float(score_arr.mean()),
        "std": float(score_arr.std()),
        "folds": scores,
    }


def calibration_points(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(prob_true, prob_pred)`` points for a calibration curve.

    Uses quantile bins so each point reflects a comparable number of samples,
    which is steadier than uniform bins under heavy class imbalance.

    Args:
        y_true: True binary labels.
        y_score: Predicted positive-class probabilities.
        n_bins: Number of calibration bins.

    Returns:
        Arrays ``(prob_true, prob_pred)`` — the observed frequency and mean
        predicted probability per bin.
    """
    prob_true, prob_pred = calibration_curve(
        np.asarray(y_true).astype(int),
        np.asarray(y_score, dtype=float),
        n_bins=n_bins,
        strategy="quantile",
    )
    return prob_true, prob_pred


def plot_calibration(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
) -> Path:
    """Plot one or more calibration curves and save to ``path``.

    Args:
        curves: Mapping of model name -> ``(prob_true, prob_pred)`` points.
        path: Destination image path (created parents).

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    for name, (prob_true, prob_pred) in curves.items():
        ax.plot(prob_pred, prob_true, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curve")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_shap_summary(
    model: Any,
    features: pd.DataFrame,
    path: Path,
    *,
    max_samples: int = 500,
    seed: int = 0,
) -> Path:
    """Save a SHAP summary (beeswarm) plot for a fitted tree model.

    Uses :class:`shap.TreeExplainer` on a (deterministic) sample of rows. SHAP is
    imported lazily so the rest of the harness does not depend on it.

    Args:
        model: A fitted tree-based estimator (e.g. ``LGBMClassifier``).
        features: Feature matrix to explain (a sample is taken if large).
        path: Destination image path.
        max_samples: Cap on rows explained, for speed.
        seed: Seed for the row sample.

    Returns:
        The path written.
    """
    import shap  # lazy: heavy optional dependency

    sample = features
    if len(features) > max_samples:
        sample = features.sample(n=max_samples, random_state=seed)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    # Binary classifiers may return a per-class list; take the positive class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
