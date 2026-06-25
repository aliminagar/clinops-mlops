"""Baseline classifiers: logistic regression and a LightGBM gradient booster.

Both are the baselines the PyTorch challenger must beat. They are built as
*unfitted* estimators so the training stage controls fitting (on the train fold
only). Imbalance is handled by **class weighting only** — never resampling:

- Logistic regression uses ``class_weight="balanced"`` and standardizes the
  continuous numeric features (binary indicators are passed through unscaled).
- LightGBM uses ``scale_pos_weight = n_negative / n_positive`` computed from the
  **train fold only** by the caller.
"""

from __future__ import annotations

from collections.abc import Sequence

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_regression(
    continuous_features: Sequence[str],
    *,
    seed: int,
    max_iter: int = 1000,
) -> Pipeline:
    """Build an unfitted, class-balanced logistic-regression pipeline.

    The pipeline standardizes the given continuous features and passes the
    remaining (binary indicator) columns through unchanged, then fits a
    class-balanced logistic regression. The scaler is part of the pipeline, so it
    is fit on the train fold only when the pipeline is fit — no leakage.

    Args:
        continuous_features: Names of the columns to standardize.
        seed: Random seed for the solver (reproducibility).
        max_iter: Maximum solver iterations.

    Returns:
        An unfitted scikit-learn :class:`~sklearn.pipeline.Pipeline`.
    """
    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), list(continuous_features))],
        remainder="passthrough",
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def build_lightgbm(
    scale_pos_weight: float,
    *,
    seed: int,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
) -> LGBMClassifier:
    """Build an unfitted, imbalance-weighted LightGBM classifier.

    Imbalance is handled with ``scale_pos_weight`` (typically ``neg / pos`` from
    the train fold), not resampling. Determinism is pinned via the seed and a
    single worker thread.

    Args:
        scale_pos_weight: Positive-class weight, computed from the train fold.
        seed: Random seed for reproducibility.
        n_estimators: Number of boosting rounds.
        learning_rate: Boosting learning rate.
        num_leaves: Maximum leaves per tree.

    Returns:
        An unfitted :class:`lightgbm.LGBMClassifier`.
    """
    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        verbose=-1,
    )


def scale_pos_weight_from(target: Sequence[int]) -> float:
    """Compute LightGBM's ``scale_pos_weight`` (``neg / pos``) from labels.

    Args:
        target: Binary labels of a single fold (the train fold).

    Returns:
        ``n_negative / n_positive``; falls back to ``1.0`` if there are no
        positives (degenerate fold), so callers never divide by zero.
    """
    labels = list(target)
    positives = sum(1 for value in labels if value == 1)
    negatives = len(labels) - positives
    if positives == 0:
        return 1.0
    return negatives / positives
