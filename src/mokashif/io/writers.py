"""Writers: COG (Cloud-Optimised GeoTIFF) and GeoJSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rasterio.transform import Affine


def write_cog(
    path: str | Path,
    data: np.ndarray,
    transform: Affine,
    crs: str,
    nodata: float | None = None,
    compress: str = "deflate",
) -> Path:
    """Write ``data`` as a Cloud-Optimised GeoTIFF.

    ``data`` may be 2-D (H, W) or 3-D (C, H, W).
    """
    import rasterio
    from rasterio.enums import Resampling

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[None]
    count, height, width = data.shape

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": compress,
        "BIGTIFF": "IF_SAFER",
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        # Build overviews for COG-ness (2x, 4x, 8x, 16x).
        factors = [2, 4, 8, 16]
        dst.build_overviews(factors, Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")
    return path


def write_geojson(path: str | Path, features: list[dict], crs: str = "EPSG:4326") -> Path:
    """Write a GeoJSON ``FeatureCollection`` (always re-projected to EPSG:4326)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fc = {
        "type": "FeatureCollection",
        "name": "mokashif_changes",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    path.write_text(json.dumps(fc, indent=2))
    return path
