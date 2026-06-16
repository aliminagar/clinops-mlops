"""Tests for the split/load layer (zero-variance drop + stratified splits)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clinops.config import settings
from clinops.training import split

TARGET = settings.target_name


def test_load_feature_store_drops_zero_variance_and_target(
    feature_store: pd.DataFrame, tmp_path: Path
) -> None:
    path = tmp_path / "features.parquet"
    feature_store.to_parquet(path)

    x, y = split.load_feature_store(path)
    assert TARGET not in x.columns
    assert "cond_copd" not in x.columns  # constant column auto-dropped
    assert len(x) == len(y) == len(feature_store)
    assert list(x.index) == list(y.index)


def test_drop_zero_variance_reports_dropped(feature_store: pd.DataFrame) -> None:
    reduced, dropped = split.drop_zero_variance(feature_store.drop(columns=[TARGET]))
    assert dropped == ["cond_copd"]
    assert "cond_copd" not in reduced.columns


def test_build_splits_proportions_and_stratification(feature_store: pd.DataFrame) -> None:
    y = feature_store[TARGET]
    splits = split.build_splits(y, seed=settings.random_seed)

    counts = {name: int((splits == name).sum()) for name in split.SPLIT_NAMES}
    assert sum(counts.values()) == len(y)
    # 60/20/20 within a small tolerance.
    assert abs(counts["train"] / len(y) - 0.6) < 0.02
    assert abs(counts["val"] / len(y) - 0.2) < 0.02
    assert abs(counts["test"] / len(y) - 0.2) < 0.02

    # No row is in two splits; every split sees both classes (stratified).
    assert splits.notna().all()
    for name in split.SPLIT_NAMES:
        labels = y[splits == name]
        assert labels.nunique() == 2


def test_build_splits_is_deterministic(feature_store: pd.DataFrame) -> None:
    y = feature_store[TARGET]
    a = split.build_splits(y, seed=7)
    b = split.build_splits(y, seed=7)
    pd.testing.assert_series_equal(a, b)
    # A different seed yields a different partition.
    c = split.build_splits(y, seed=8)
    assert not a.equals(c)


def test_get_or_create_splits_persists_and_reuses(
    feature_store: pd.DataFrame, tmp_path: Path
) -> None:
    y = feature_store[TARGET]
    path = tmp_path / "splits.parquet"

    first = split.get_or_create_splits(y, path=path, seed=settings.random_seed)
    assert path.exists()
    second = split.get_or_create_splits(y, path=path, seed=settings.random_seed)
    pd.testing.assert_series_equal(first, second)


def test_split_xy_partitions_align(feature_store: pd.DataFrame) -> None:
    y = feature_store[TARGET]
    x = feature_store.drop(columns=[TARGET])
    splits = split.build_splits(y, seed=settings.random_seed)

    x_tr, y_tr = split.split_xy(x, y, splits, "train")
    assert list(x_tr.index) == list(y_tr.index)
    assert (splits.loc[x_tr.index] == "train").all()
