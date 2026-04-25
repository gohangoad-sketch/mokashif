"""Tests for normalisation strategies."""

from __future__ import annotations

import numpy as np
import pytest

from mokashif.io.readers import RasterImage
from mokashif.preprocess.normalize import normalize_image


@pytest.fixture
def img() -> RasterImage:
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4096, size=(3, 32, 32)).astype(np.float32)
    from rasterio.transform import from_origin

    return RasterImage(
        data=data,
        transform=from_origin(0, 0, 1, 1),
        crs="EPSG:32633",
    )


def test_percentile_normalises_to_unit_range(img: RasterImage) -> None:
    out = normalize_image(img, method="percentile", p_low=2, p_high=98)
    assert out.data.min() >= 0.0
    assert out.data.max() <= 1.0
    assert 0.05 < out.data.mean() < 0.95


def test_zscore_has_unit_std(img: RasterImage) -> None:
    out = normalize_image(img, method="zscore")
    for c in range(out.data.shape[0]):
        assert abs(out.data[c].mean()) < 1e-3
        assert abs(out.data[c].std() - 1.0) < 1e-3


def test_none_is_passthrough(img: RasterImage) -> None:
    out = normalize_image(img, method="none")
    np.testing.assert_array_equal(out.data, img.data)


def test_unknown_method_raises(img: RasterImage) -> None:
    with pytest.raises(ValueError):
        normalize_image(img, method="bogus")
