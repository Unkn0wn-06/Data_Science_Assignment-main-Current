"""Pure row-aligned hard and soft expert routing functions."""

from __future__ import annotations

import numpy as np


def _aligned(*arrays):
    values = [np.asarray(array, dtype=float) for array in arrays]
    if len({len(value) for value in values}) != 1:
        raise ValueError("Routing arrays must have identical lengths.")
    return values


def hard_route(standard_prediction, premium_prediction, premium_probability, threshold):
    standard, premium, probability = _aligned(
        standard_prediction, premium_prediction, premium_probability
    )
    routed = probability >= float(threshold)
    return np.where(routed, premium, standard), routed


def soft_route(standard_prediction, premium_prediction, premium_probability):
    standard, premium, probability = _aligned(
        standard_prediction, premium_prediction, premium_probability
    )
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Soft-routing probabilities must lie in [0, 1].")
    return (1.0 - probability) * standard + probability * premium
