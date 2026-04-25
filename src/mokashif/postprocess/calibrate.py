"""Histogram-based confidence calibration.

The raw similarity-derived probability is well-ordered but not necessarily
*calibrated* (a value of 0.5 doesn't mean 50% precision). We fit a simple
monotonic mapping using the empirical CDF of the probability map: each value
is replaced by its quantile rank, then optionally squashed by a logistic to
push easy positives / negatives toward 0/1.
"""

from __future__ import annotations

import numpy as np


def calibrate_probabilities(prob: np.ndarray, sharpness: float = 6.0) -> np.ndarray:
    """Rank-transform + logistic squash. Returns array in [0, 1]."""
    flat = prob.ravel()
    if flat.size == 0:
        return prob

    # Empirical CDF via argsort (O(N log N) but fine for tile-sized arrays).
    order = np.argsort(flat, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(flat.size)
    quantiles = ranks.astype(np.float32) / max(flat.size - 1, 1)

    # Logistic centered on 0.5 to sharpen.
    centered = quantiles - 0.5
    squashed = 1.0 / (1.0 + np.exp(-sharpness * centered))
    return squashed.reshape(prob.shape).astype(np.float32)
