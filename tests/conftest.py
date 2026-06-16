"""Shared fixtures: a small synthetic feature store (no real data required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clinops.config import settings


def make_synthetic_store(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic feature store mirroring the real schema.

    Includes continuous and binary features, a constant (zero-variance) column
    (``cond_copd``) that must be auto-dropped, and an imbalanced target that
    genuinely depends on a few features so models score above the base rate.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 90, n)
    prior_inpatient = rng.poisson(0.7, n)
    chronic = rng.integers(0, 4, n)
    gender_male = rng.integers(0, 2, n)

    logit = -3.0 + 0.03 * (age - 55) + 0.6 * prior_inpatient + 0.4 * chronic
    proba = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.uniform(0, 1, n) < proba).astype(int)

    frame = pd.DataFrame(
        {
            "age": age,
            "bmi": rng.uniform(18, 45, n),
            "systolic_bp": rng.uniform(90, 180, n),
            "prior_encounter_count": rng.poisson(20, n),
            "prior_inpatient_count": prior_inpatient,
            "chronic_condition_count": chronic,
            "gender_male": gender_male,
            "gender_female": 1 - gender_male,
            "cond_diabetes": rng.integers(0, 2, n),
            "cond_copd": np.zeros(n, dtype=int),  # constant -> zero variance
            settings.target_name: target,
        },
        index=pd.Index([f"p{i:04d}" for i in range(n)], name="patient_id"),
    )
    return frame


@pytest.fixture
def feature_store() -> pd.DataFrame:
    """A deterministic synthetic feature store with a usable positive rate."""
    frame = make_synthetic_store()
    assert int(frame[settings.target_name].sum()) > 20  # enough positives to stratify
    return frame
