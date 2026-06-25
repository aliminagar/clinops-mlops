"""PyTorch feed-forward challenger, wrapped in a scikit-learn-compatible API.

A small feed-forward neural network competes against the sklearn + LightGBM
baselines on the **same** features, CV protocol, and split. :class:`TorchMLPClassifier`
exposes ``fit(X, y)`` / ``predict_proba(X)`` so it is a drop-in for the existing
cross-validation, threshold-selection, and MLflow logging flow.

Design choices (intentional):

- **Scaling lives inside the wrapper.** Unlike the trees, a neural net needs
  standardized inputs, so a :class:`~sklearn.preprocessing.StandardScaler` is fit
  on the training fold *inside* ``fit`` (no leakage). The linear/tree baselines do
  not standardize; that asymmetry is deliberate and per-model.
- **Imbalance via ``pos_weight``.** ``BCEWithLogitsLoss`` is weighted by
  ``pos_weight = n_neg / n_pos`` from the training fold (~3.5% prevalence),
  mirroring LightGBM's ``scale_pos_weight``.
- **Determinism.** Torch and NumPy are seeded from ``settings.random_seed`` and
  training is full-batch (no shuffling), so repeated fits on one machine are
  reproducible. Residual nondeterminism (BLAS/thread scheduling, hardware) can
  still cause tiny float differences across machines — exact bit-equality is not
  guaranteed off this CPU.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from clinops.config import settings

logger = logging.getLogger(__name__)


class FeedForwardClassifier(nn.Module):
    """Feed-forward binary classifier emitting a single logit per row."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        """Initialize a two-hidden-layer MLP.

        Args:
            input_dim: Number of input features.
            hidden_dim: Width of the first hidden layer (second is half).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass and return logits of shape ``(n, 1)``."""
        logits: torch.Tensor = self.net(x)
        return logits


class TorchMLPClassifier:
    """scikit-learn-compatible feed-forward NN challenger (CPU, deterministic).

    Implements just enough of the estimator API (``fit`` / ``predict_proba`` /
    ``predict`` / ``classes_``) to be a drop-in for the shared CV + threshold +
    MLflow pipeline. See the module docstring for the scaling / imbalance /
    determinism rationale.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        hidden_dim: int = 64,
        epochs: int = 400,
        lr: float = 5e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        """Configure the challenger.

        Args:
            seed: Seed for torch/numpy; defaults to ``settings.random_seed``.
            hidden_dim: Width of the first hidden layer.
            epochs: Number of full-batch training epochs.
            lr: Adam learning rate.
            weight_decay: Adam L2 weight decay.
        """
        self.seed = settings.random_seed if seed is None else seed
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.scaler = StandardScaler()
        self.classes_ = np.array([0, 1])
        self.model_: FeedForwardClassifier | None = None
        self.pos_weight_: float | None = None
        self.feature_names_: list[str] | None = None

    def _seed_everything(self) -> None:
        """Seed torch + numpy for reproducible weight init and training."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def fit(self, x: Any, y: Any) -> TorchMLPClassifier:
        """Fit the network on standardized inputs with imbalance weighting.

        Args:
            x: Feature matrix (DataFrame or array) for the training fold.
            y: Binary target aligned with ``x``.

        Returns:
            ``self`` (fitted).
        """
        self._seed_everything()
        self.feature_names_ = list(x.columns) if isinstance(x, pd.DataFrame) else None

        x_scaled = self.scaler.fit_transform(np.asarray(x, dtype=np.float64))
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        n_pos = float(y_arr.sum())
        n_neg = float(len(y_arr) - n_pos)
        self.pos_weight_ = n_neg / n_pos if n_pos > 0 else 1.0

        x_t = torch.tensor(x_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)
        pos_weight = torch.tensor([self.pos_weight_], dtype=torch.float32)

        model = FeedForwardClassifier(x_scaled.shape[1], self.hidden_dim)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(model(x_t), y_t)
            loss.backward()
            optimizer.step()
        model.eval()

        self.model_ = model
        logger.debug(
            "TorchMLPClassifier fit: n=%d, pos_weight=%.2f, final_loss=%.4f",
            len(y_arr),
            self.pos_weight_,
            float(loss.item()),
        )
        return self

    def predict_proba(self, x: Any) -> np.ndarray:
        """Return class probabilities ``[P(0), P(1)]`` for each row.

        Args:
            x: Feature matrix (DataFrame or array).

        Returns:
            Array of shape ``(n, 2)`` with calibrated probabilities in ``[0, 1]``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.model_ is None:
            raise RuntimeError("TorchMLPClassifier is not fitted; call fit() first.")
        x_scaled = self.scaler.transform(np.asarray(x, dtype=np.float64))
        x_t = torch.tensor(x_scaled, dtype=torch.float32)
        with torch.no_grad():
            positive = torch.sigmoid(self.model_(x_t)).numpy().reshape(-1)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, x: Any) -> np.ndarray:
        """Return hard class labels at a 0.5 probability cutoff."""
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)
