"""Flavor-agnostic MLflow ``pyfunc`` wrapper returning positive-class probability.

Every candidate model (logistic regression, LightGBM, the PyTorch challenger) is
logged through :class:`ProbaModel`. That makes the serving layer framework-blind:
``mlflow.pyfunc.load_model(uri).predict(df)`` returns the positive-class
probability regardless of which flavor won promotion, so a champion of any type
serves through the same ``/predict`` path.

This module deliberately imports only MLflow + numpy/pandas (no BentoML, no
Torch) so the training stage can log through it without pulling serving deps, and
it stays importable wherever the pyfunc model is later unpickled.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel


class ProbaModel(PythonModel):
    """Wrap any ``predict_proba`` estimator so pyfunc yields ``P(positive)``."""

    def __init__(self, estimator: Any) -> None:
        """Store the fitted estimator (serialized with the logged model)."""
        self.estimator = estimator

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return the positive-class probability for each input row.

        Args:
            context: MLflow python-model context (unused).
            model_input: Feature rows; columns are aligned to the logged signature.
            params: Optional inference params (unused).

        Returns:
            A 1-D array of positive-class probabilities.
        """
        proba = self.estimator.predict_proba(model_input)
        return np.asarray(proba)[:, 1]
