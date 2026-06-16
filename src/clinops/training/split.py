"""Load the feature store and build reproducible, stratified data splits.

This module is the single entry point for turning
``data/processed/features.parquet`` into model-ready ``X``/``y`` and a fixed
train/val/test partition that is **identical across every model and every run**:

- :func:`load_feature_store` reads the parquet, separates the target, and drops
  zero-variance feature columns (auto-detected, logged) so constant features such
  as ``cond_copd`` never reach a model.
- :func:`get_or_create_splits` builds a 60/20/20 split, **stratified on the
  target** with ``settings.random_seed``, and persists the per-patient split
  labels to parquet. Subsequent runs reload the saved labels, so the logistic
  regression and the LightGBM baseline are scored on exactly the same rows.

No scaling, encoding, or class weighting happens here — those are fit on the
train fold only, downstream, to avoid leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from clinops.config import settings

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")
_SPLIT_COLUMN = "split"


def drop_zero_variance(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop constant (zero-variance) feature columns.

    A column is zero-variance if it has at most one distinct value (NaNs
    included); such columns carry no signal and are removed before modelling.

    Args:
        features: The feature matrix.

    Returns:
        A tuple of (reduced feature matrix, sorted list of dropped column names).
    """
    dropped = sorted(col for col in features.columns if features[col].nunique(dropna=False) <= 1)
    if dropped:
        logger.info("Dropping %d zero-variance feature(s): %s", len(dropped), ", ".join(dropped))
    reduced = features.drop(columns=dropped)
    return reduced, dropped


def load_feature_store(
    path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the feature store into ``X``/``y`` with zero-variance columns dropped.

    Args:
        path: Path to the feature-store parquet; defaults to
            ``settings.data_processed / "features.parquet"``.

    Returns:
        A tuple of (feature matrix ``X`` with zero-variance columns removed,
        target vector ``y``). ``X`` keeps the ``patient_id`` index; the target
        column never appears in ``X``.

    Raises:
        ValueError: If the target column is missing from the feature store.
    """
    source = Path(path) if path is not None else settings.data_processed / "features.parquet"
    frame = pd.read_parquet(source)

    target = settings.target_name
    if target not in frame.columns:
        raise ValueError(f"Feature store {source!s} is missing target column {target!r}.")

    y = frame[target].astype(int)
    x = frame.drop(columns=[target])
    x, _ = drop_zero_variance(x)
    logger.info("Loaded feature store: X shape=(%d, %d) from %s", x.shape[0], x.shape[1], source)
    return x, y


def build_splits(
    target: pd.Series,
    *,
    seed: int,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> pd.Series:
    """Assign each row to ``train``/``val``/``test``, stratified on the target.

    The split is two stratified cuts: first ``test_size`` is held out, then
    ``val_size`` (as a fraction of the whole) is taken from the remainder; what
    is left is the train fold. The default 0.2/0.2 yields a 60/20/20 split.

    Args:
        target: The binary target, indexed by ``patient_id``.
        seed: Random seed controlling the (reproducible) split.
        val_size: Validation fraction of the full dataset.
        test_size: Test fraction of the full dataset.

    Returns:
        A ``str`` Series aligned to ``target.index`` with values in
        :data:`SPLIT_NAMES`.
    """
    # Split on integer positions (not the patient_id index) so the routine is
    # agnostic to the index dtype, e.g. pandas' pyarrow-backed string index.
    labels_arr = target.to_numpy()
    positions = np.arange(len(target))
    train_val_pos, test_pos = train_test_split(
        positions,
        test_size=test_size,
        stratify=labels_arr,
        random_state=seed,
    )
    # val_size is a fraction of the whole; convert to a fraction of train_val.
    relative_val = val_size / (1.0 - test_size)
    train_pos, val_pos = train_test_split(
        train_val_pos,
        test_size=relative_val,
        stratify=labels_arr[train_val_pos],
        random_state=seed,
    )

    split_labels = np.empty(len(target), dtype=object)
    split_labels[train_pos] = "train"
    split_labels[val_pos] = "val"
    split_labels[test_pos] = "test"
    return pd.Series(split_labels, index=target.index, name=_SPLIT_COLUMN)


def save_splits(splits: pd.Series, path: Path) -> None:
    """Persist per-patient split labels to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_frame().to_parquet(path)


def load_splits(path: Path) -> pd.Series:
    """Load persisted per-patient split labels from parquet."""
    return pd.read_parquet(path)[_SPLIT_COLUMN]


def get_or_create_splits(
    target: pd.Series,
    *,
    path: Path | None = None,
    seed: int | None = None,
) -> pd.Series:
    """Load persisted splits if present, otherwise build and persist them.

    Reusing the saved labels guarantees every model is trained and scored on the
    same rows. If the persisted labels do not cover exactly ``target.index`` (e.g.
    the cohort was regenerated), the splits are rebuilt.

    Args:
        target: The binary target, indexed by ``patient_id``.
        path: Where the split labels live; defaults to
            ``settings.data_processed / "splits.parquet"``.
        seed: Random seed; defaults to ``settings.random_seed``.

    Returns:
        The per-patient split-label Series aligned to ``target.index``.
    """
    destination = Path(path) if path is not None else settings.data_processed / "splits.parquet"
    resolved_seed = settings.random_seed if seed is None else seed

    if destination.exists():
        existing = load_splits(destination)
        if existing.index.symmetric_difference(target.index).empty:
            logger.info("Reusing persisted splits from %s", destination)
            return existing.reindex(target.index)
        logger.warning("Persisted splits at %s do not match the cohort; rebuilding.", destination)

    splits = build_splits(target, seed=resolved_seed)
    save_splits(splits, destination)
    counts = {name: int((splits == name).sum()) for name in SPLIT_NAMES}
    logger.info("Built stratified splits %s and saved to %s", counts, destination)
    return splits


def split_xy(
    features: pd.DataFrame,
    target: pd.Series,
    splits: pd.Series,
    name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the ``(X, y)`` subset for a single split partition.

    Args:
        features: The full feature matrix.
        target: The full target vector.
        splits: Per-row split labels.
        name: One of :data:`SPLIT_NAMES`.

    Returns:
        The ``(X, y)`` rows whose split label equals ``name``.
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {name!r}; expected one of {SPLIT_NAMES}.")
    mask = splits == name
    return features.loc[mask], target.loc[mask]
