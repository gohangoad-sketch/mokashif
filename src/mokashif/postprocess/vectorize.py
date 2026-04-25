"""Turn probability maps into clean polygon features."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rasterio.transform import Affine


def threshold_and_clean(
    prob: np.ndarray,
    threshold: float = 0.5,
    morphology_kernel: int = 3,
    min_pixels: int = 16,
) -> np.ndarray:
    """Binarise a probability map and clean up speckle.

    Returns an ``np.uint8`` mask (0/1).
    """
    from scipy import ndimage as ndi

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    mask = (prob >= threshold).astype(np.uint8)
    if morphology_kernel > 1:
        struct = np.ones((morphology_kernel, morphology_kernel), dtype=bool)
        mask = ndi.binary_opening(mask, structure=struct).astype(np.uint8)
        mask = ndi.binary_closing(mask, structure=struct).astype(np.uint8)
    if min_pixels > 1:
        labeled, n = ndi.label(mask)
        if n > 0:
            sizes = ndi.sum(mask, labeled, range(1, n + 1))
            small = np.where(sizes < min_pixels)[0] + 1
            for s in small:
                mask[labeled == s] = 0
    return mask


def mask_to_features(
    mask: np.ndarray,
    transform: Affine,
    crs: str,
    confidence: np.ndarray | None = None,
) -> list[dict]:
    """Convert a binary mask to GeoJSON features (re-projected to EPSG:4326)."""
    import rasterio.features
    import shapely.geometry
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    project = transformer.transform

    features: list[dict] = []
    for geom_dict, value in rasterio.features.shapes(
        mask, mask=mask.astype(bool), transform=transform
    ):
        if value == 0:
            continue
        geom = shapely.geometry.shape(geom_dict)
        if geom.is_empty:
            continue

        props: dict = {"class": "change"}
        if confidence is not None:
            mini_mask = rasterio.features.geometry_mask(
                [geom_dict],
                out_shape=mask.shape,
                transform=transform,
                invert=True,
            )
            vals = confidence[mini_mask]
            if vals.size > 0:
                props["confidence"] = float(np.median(vals))
                props["confidence_p90"] = float(np.percentile(vals, 90))
                props["area_pixels"] = int(mini_mask.sum())

        wgs84_geom = shp_transform(project, geom)
        features.append(
            {
                "type": "Feature",
                "geometry": shapely.geometry.mapping(wgs84_geom),
                "properties": props,
            }
        )
    return features
