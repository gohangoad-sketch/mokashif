"""Raster readers that work with **any** sensor format rasterio can open.

We deliberately keep this module dependency-light:
- ``rasterio`` does the heavy lifting (GDAL under the hood => GeoTIFF, COG, JP2,
  NITF, HDF5, NetCDF, Zarr, …).
- We expose a single ``RasterPair`` dataclass that other stages consume; this
  decouples the rest of the pipeline from on-disk formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rasterio.transform import Affine


@dataclass
class RasterImage:
    """A single raster in memory, with all the geo-metadata we care about."""

    data: np.ndarray
    """Image array, shape (C, H, W), float32 in [0, 1] after preprocessing."""

    transform: Affine
    """Affine transform from pixel to world coordinates."""

    crs: str
    """CRS as a string (e.g. ``"EPSG:32633"``)."""

    nodata: float | None = None
    """Nodata sentinel for masking, if any."""

    source_path: Path | None = None
    """Original file path, for debugging / provenance."""

    band_descriptions: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.data.shape[-2])

    @property
    def width(self) -> int:
        return int(self.data.shape[-1])

    @property
    def channels(self) -> int:
        return int(self.data.shape[0]) if self.data.ndim == 3 else 1

    @property
    def shape(self) -> tuple[int, int, int]:
        h, w = self.height, self.width
        return (self.channels, h, w)


@dataclass
class RasterPair:
    """A spatially aligned pair of rasters (before, after)."""

    before: RasterImage
    after: RasterImage

    def __post_init__(self) -> None:
        if self.before.shape[1:] != self.after.shape[1:]:
            raise ValueError(
                f"before/after spatial shapes differ: "
                f"{self.before.shape[1:]} vs {self.after.shape[1:]}"
            )
        if self.before.crs != self.after.crs:
            raise ValueError(f"before/after CRS differ: {self.before.crs} vs {self.after.crs}")


def _select_rgb(rio_dataset) -> list[int]:
    """Pick three indices that look most like an RGB triple.

    Strategy: prefer bands explicitly tagged "Red"/"Green"/"Blue"; otherwise
    use the first three bands. This is sensor-agnostic.
    """
    descriptions = [(d or "").lower() for d in (rio_dataset.descriptions or [])]
    color_interp = [str(c).lower() for c in (rio_dataset.colorinterp or [])]
    n = rio_dataset.count

    # Try by colorinterp tag first (set by GDAL for many formats).
    interp_to_idx: dict[str, int] = {}
    for i, c in enumerate(color_interp, start=1):
        for key in ("red", "green", "blue"):
            if key in c and key not in interp_to_idx:
                interp_to_idx[key] = i
    if all(k in interp_to_idx for k in ("red", "green", "blue")):
        return [interp_to_idx["red"], interp_to_idx["green"], interp_to_idx["blue"]]

    # Then try descriptions.
    desc_to_idx: dict[str, int] = {}
    for i, d in enumerate(descriptions, start=1):
        for key in ("red", "green", "blue"):
            if key in d and key not in desc_to_idx:
                desc_to_idx[key] = i
    if all(k in desc_to_idx for k in ("red", "green", "blue")):
        return [desc_to_idx["red"], desc_to_idx["green"], desc_to_idx["blue"]]

    # Fallback: first three bands. If only one band, repeat it.
    if n >= 3:
        return [1, 2, 3]
    return [1] * 3


def open_raster(
    path: str | Path,
    bands: list[int] | None = None,
    as_rgb: bool = True,
) -> RasterImage:
    """Read a raster from disk into a :class:`RasterImage`.

    Parameters
    ----------
    path
        File path readable by GDAL/rasterio.
    bands
        Explicit 1-based band indices. If ``None`` and ``as_rgb`` is True we
        auto-select an RGB triple.
    as_rgb
        Auto-pick three RGB bands when ``bands`` is None.
    """
    import rasterio  # local import keeps the module importable without rasterio for type-only use

    path = Path(path)
    with rasterio.open(path) as src:
        if bands is None:
            bands = _select_rgb(src) if as_rgb else list(range(1, src.count + 1))
        arr = src.read(bands).astype(np.float32, copy=False)
        descriptions = [(src.descriptions[b - 1] or f"band_{b}") for b in bands]
        return RasterImage(
            data=arr,
            transform=src.transform,
            crs=str(src.crs) if src.crs else "EPSG:4326",
            nodata=src.nodata,
            source_path=path,
            band_descriptions=descriptions,
        )


def prepare_pair(
    before: str | Path,
    after: str | Path,
    bands: list[int] | None = None,
) -> RasterPair:
    """Convenience: open *before* and *after* with matching band selection.

    The two files are assumed to already share a CRS and grid; if they don't,
    use :func:`mokashif.preprocess.align.align_pair` first.
    """
    b = open_raster(before, bands=bands)
    a = open_raster(after, bands=bands)
    return RasterPair(before=b, after=a)
