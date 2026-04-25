"""Per-channel normalisation strategies.

We support three:

- ``percentile``  - robust [p_low, p_high] -> [0, 1] (default; works on any DN range).
- ``zscore``      - subtract mean, divide by std (good for SAR / log-amp).
- ``none``        - leave the array alone (caller did it).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from mokashif.io.readers import RasterImage, RasterPair


def normalize_image(
    image: RasterImage,
    method: str = "percentile",
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> RasterImage:
    arr = image.data.astype(np.float32, copy=True)
    nodata = image.nodata
    valid = np.ones_like(arr, dtype=bool)
    if nodata is not None:
        valid = arr != nodata

    if method == "none":
        return image

    if method == "percentile":
        out = np.zeros_like(arr)
        for c in range(arr.shape[0]):
            band = arr[c]
            mask = valid[c] if valid.ndim == 3 else valid
            vals = band[mask] if mask.any() else band.ravel()
            if vals.size == 0:
                continue
            lo = np.percentile(vals, p_low)
            hi = np.percentile(vals, p_high)
            denom = max(hi - lo, 1e-6)
            out[c] = np.clip((band - lo) / denom, 0.0, 1.0)
        return replace(image, data=out)

    if method == "zscore":
        out = np.zeros_like(arr)
        for c in range(arr.shape[0]):
            band = arr[c]
            mask = valid[c] if valid.ndim == 3 else valid
            vals = band[mask] if mask.any() else band.ravel()
            if vals.size == 0:
                continue
            mu = float(vals.mean())
            sigma = float(vals.std()) or 1e-6
            out[c] = (band - mu) / sigma
        return replace(image, data=out)

    raise ValueError(f"Unknown normalize method: {method}")


def normalize_pair(
    pair: RasterPair,
    method: str = "percentile",
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> RasterPair:
    return RasterPair(
        before=normalize_image(pair.before, method, p_low, p_high),
        after=normalize_image(pair.after, method, p_low, p_high),
    )
