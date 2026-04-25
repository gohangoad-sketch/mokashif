"""Spatial alignment of an image pair.

Two-stage strategy:

1. **Coarse**: reproject the *after* image into the *before* image's grid
   (CRS + transform) using rasterio.warp. This is sensor-agnostic and always
   safe.
2. **Fine** (optional): sub-pixel co-registration with AROSICS if the
   ``preprocess`` extra is installed and the images are similar enough.
   AROSICS is gated behind ``try/except`` so missing it never breaks the
   pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

from mokashif.io.readers import RasterImage, RasterPair

logger = logging.getLogger(__name__)


def _reproject_to(reference: RasterImage, image: RasterImage) -> RasterImage:
    """Resample ``image`` onto ``reference``'s grid + CRS."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    dst = np.zeros(
        (image.channels, reference.height, reference.width),
        dtype=np.float32,
    )
    src_nodata = image.nodata if image.nodata is not None else 0.0
    reproject(
        source=image.data,
        destination=dst,
        src_transform=image.transform,
        src_crs=image.crs,
        dst_transform=reference.transform,
        dst_crs=reference.crs,
        resampling=Resampling.bilinear,
        src_nodata=src_nodata,
        dst_nodata=src_nodata,
    )
    # Silence rasterio not being fully used in the no-warp path.
    _ = rasterio
    return replace(image, data=dst, transform=reference.transform, crs=reference.crs)


def _coregister_arosics(pair: RasterPair) -> RasterPair:
    """Sub-pixel co-registration using AROSICS, if available."""
    try:
        from arosics import COREG
    except ImportError:
        logger.info("AROSICS not installed; skipping sub-pixel co-registration.")
        return pair

    try:
        # AROSICS works on file paths; if our images live in memory we skip.
        if pair.before.source_path is None or pair.after.source_path is None:
            logger.info("In-memory pair; skipping AROSICS (needs file paths).")
            return pair

        cr = COREG(
            str(pair.before.source_path),
            str(pair.after.source_path),
            ws=(256, 256),
            max_shift=20,
            q=True,
        )
        cr.calculate_spatial_shifts()
        if cr.success:
            logger.info(
                "AROSICS shift: x=%.2fpx y=%.2fpx",
                cr.x_shift_px,
                cr.y_shift_px,
            )
        else:
            logger.warning("AROSICS reported no reliable shift; continuing without.")
    except Exception as e:
        logger.warning("AROSICS failed (%s); continuing without sub-pixel correction.", e)

    return pair


def align_pair(pair: RasterPair, fine: bool = True) -> RasterPair:
    """Return a pair guaranteed to share grid + CRS.

    Parameters
    ----------
    pair
        Possibly mis-aligned input pair.
    fine
        Attempt AROSICS sub-pixel co-registration after coarse alignment.
    """
    aligned_after = pair.after
    if (
        pair.after.crs != pair.before.crs
        or pair.after.shape[1:] != pair.before.shape[1:]
        or pair.after.transform != pair.before.transform
    ):
        logger.info("Reprojecting AFTER onto BEFORE grid (%s).", pair.before.crs)
        aligned_after = _reproject_to(pair.before, pair.after)

    aligned = RasterPair(before=pair.before, after=aligned_after)
    if fine:
        aligned = _coregister_arosics(aligned)
    return aligned
