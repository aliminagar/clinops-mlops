"""Champion/challenger promotion logic against the MLflow model registry.

Compares the evaluated candidates on the configured promotion metric and decides
whether a challenger should be promoted to champion. The registry is the single
source of truth for which model BentoML serves.
"""

from __future__ import annotations


def promote_best(metrics: dict[str, dict[str, float]]) -> str:
    """Promote the best model to champion in the MLflow registry.

    Args:
        metrics: Mapping of model name -> metric name -> value (from evaluate).

    Returns:
        The registry version/identifier of the newly promoted champion.
    """
    raise NotImplementedError
