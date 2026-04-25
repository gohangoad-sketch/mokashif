"""Shared test fixtures: synthetic before/after pairs that don't need network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def synthetic_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two RGB GeoTIFFs identical except for a square 'change' on the right.

    We use rasterio to write proper geotiffs so the full pipeline (reproject,
    align, vectorise) can run end-to-end.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    H, W = 256, 256
    rng = np.random.default_rng(42)
    base = rng.integers(50, 200, size=(3, H, W), dtype=np.uint8)

    before = base.copy()
    after = base.copy()
    # Inject a 64x64 bright square at (96:160, 96:160) in the *after* image.
    after[:, 96:160, 96:160] = 255

    transform = from_origin(west=500_000.0, north=4_000_000.0, xsize=1.0, ysize=1.0)
    crs = "EPSG:32633"
    profile = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": 3,
        "dtype": "uint8",
        "transform": transform,
        "crs": crs,
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
    }

    bp = tmp_path / "before.tif"
    ap = tmp_path / "after.tif"
    with rasterio.open(bp, "w", **profile) as dst:
        dst.write(before)
    with rasterio.open(ap, "w", **profile) as dst:
        dst.write(after)
    return bp, ap


@pytest.fixture
def small_pair_arrays() -> tuple[np.ndarray, np.ndarray]:
    """In-memory pair without rasterio dependency."""
    rng = np.random.default_rng(123)
    base = rng.random((3, 64, 64), dtype=np.float32)
    after = base.copy()
    after[:, 32:48, 32:48] = 1.0
    return base, after
