"""Tests for confidence calibration."""

from __future__ import annotations

import numpy as np

from mokashif.postprocess.calibrate import calibrate_probabilities


def test_calibration_preserves_order() -> None:
    rng = np.random.default_rng(0)
    arr = rng.random((50, 50), dtype=np.float32)
    out = calibrate_probabilities(arr)
    # Orders must match exactly (rank-preserving transform).
    assert (np.argsort(arr, axis=None) == np.argsort(out, axis=None)).all()


def test_calibration_outputs_in_unit_range() -> None:
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(64, 64)).astype(np.float32)
    out = calibrate_probabilities(arr)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_calibration_handles_empty() -> None:
    arr = np.array([], dtype=np.float32).reshape(0, 0)
    out = calibrate_probabilities(arr)
    assert out.shape == (0, 0)
