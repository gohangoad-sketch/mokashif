"""Top-level pipeline glue: ``detect_changes(before, after, ...) -> result``.

This is the entry point we expose at the top of the package and from the CLI.
It is intentionally orchestrator-agnostic (no Prefect / Dagster) so the same
function works in a notebook, a script, or inside a flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from mokashif.config import PipelineConfig
from mokashif.io.readers import RasterPair, prepare_pair
from mokashif.io.writers import write_cog, write_geojson
from mokashif.models.change_detector import SimilarityChangeDetector
from mokashif.models.encoders import build_encoder
from mokashif.postprocess.calibrate import calibrate_probabilities
from mokashif.postprocess.vectorize import mask_to_features, threshold_and_clean
from mokashif.preprocess.align import align_pair
from mokashif.preprocess.normalize import normalize_pair
from mokashif.tiling.window import Tile, generate_tiles, iterate_tile_pairs, stitch_tiles

logger = logging.getLogger(__name__)


@dataclass
class ChangeDetectionResult:
    """All outputs of a pipeline run."""

    probability: np.ndarray
    """(H, W) float32 probability map (0 = unchanged, 1 = changed)."""

    mask: np.ndarray
    """(H, W) uint8 binary mask after thresholding + cleanup."""

    features: list[dict] = field(default_factory=list)
    """GeoJSON features (in EPSG:4326)."""

    transform: object | None = None
    """Affine transform of the reference grid."""

    crs: str | None = None
    """CRS of the reference grid."""

    config: PipelineConfig | None = None
    """The exact config used for this run, for provenance."""

    def summary(self) -> str:
        n = len(self.features)
        pct = 0.0 if self.probability.size == 0 else float((self.mask > 0).mean() * 100.0)
        return (
            f"Mokashif change-detection result: {n} polygon(s); "
            f"{pct:.2f}% of pixels marked as changed."
        )

    def to_geojson(self, path: str | Path) -> Path:
        return write_geojson(path, self.features)

    def write_outputs(self, output_dir: str | Path) -> dict[str, Path]:
        """Write probability COG, mask COG and GeoJSON to ``output_dir``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.transform is None or self.crs is None:
            raise RuntimeError("Result is missing geo-metadata; cannot write outputs.")

        prob_path = write_cog(
            output_dir / "confidence.tif",
            (self.probability * 10_000).astype(np.uint16),
            self.transform,
            self.crs,
            nodata=None,
        )
        mask_path = write_cog(
            output_dir / "change_mask.tif",
            self.mask.astype(np.uint8),
            self.transform,
            self.crs,
            nodata=0,
        )
        gj_path = self.to_geojson(output_dir / "changes.geojson")
        return {"confidence": prob_path, "mask": mask_path, "geojson": gj_path}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _run_detector_tiled(
    detector: SimilarityChangeDetector,
    pair: RasterPair,
    config: PipelineConfig,
    device: torch.device,
) -> np.ndarray:
    h, w = pair.before.shape[1], pair.before.shape[2]
    tiles: list[Tile] = generate_tiles(
        h,
        w,
        tile_size=config.tiling.tile_size,
        overlap=config.tiling.overlap,
    )
    logger.info("Generated %d tiles for %dx%d image.", len(tiles), h, w)

    outputs: list[np.ndarray] = []
    for _tile, b_patch, a_patch in iterate_tile_pairs(
        pair.before.data, pair.after.data, tiles, pad_value=config.tiling.pad_value
    ):
        prob = detector.detect(b_patch, a_patch, device=device)
        outputs.append(prob)

    return stitch_tiles(h, w, tiles, outputs)


def detect_changes(
    before: str | Path | RasterPair,
    after: str | Path | None = None,
    config: PipelineConfig | None = None,
    device: str | torch.device = "auto",
    output_dir: str | Path | None = None,
) -> ChangeDetectionResult:
    """Run the full pipeline.

    Parameters
    ----------
    before, after
        Paths to two rasters. Alternatively, pass a single :class:`RasterPair`
        as ``before`` (and leave ``after=None``) to skip I/O.
    config
        Optional :class:`PipelineConfig`. Defaults are sensible for SOTA RGB
        change detection.
    device
        ``"cpu"``, ``"cuda"``, ``"mps"`` or ``"auto"``.
    output_dir
        If provided, writes ``confidence.tif``, ``change_mask.tif`` and
        ``changes.geojson`` here.
    """
    config = config or PipelineConfig()
    dev = _resolve_device(str(device) if not isinstance(device, str) else device)

    if isinstance(before, RasterPair):
        pair = before
    else:
        if after is None:
            raise ValueError("Provide both `before` and `after` paths (or a RasterPair).")
        pair = prepare_pair(before, after)

    logger.info("Aligning pair (%s).", "fine+coarse" if config.preprocess.coregister else "coarse")
    pair = align_pair(pair, fine=config.preprocess.coregister)
    pair = normalize_pair(
        pair,
        method=config.preprocess.normalize,
        p_low=config.preprocess.percentile_low,
        p_high=config.preprocess.percentile_high,
    )

    encoder = build_encoder(config.model.encoder, weights_dir=config.model.weights_dir)
    detector = SimilarityChangeDetector(encoder, similarity=config.model.similarity)

    prob = _run_detector_tiled(detector, pair, config, dev)
    if config.postprocess.calibrate:
        prob = calibrate_probabilities(prob)

    # Min-area in pixels (using before image GSD if known; else 1 px = 1 m).
    pixel_area_m2 = abs(pair.before.transform.a * pair.before.transform.e)
    if pixel_area_m2 <= 0:
        pixel_area_m2 = 1.0
    min_pixels = max(int(config.postprocess.min_object_area_m2 / pixel_area_m2), 1)

    mask = threshold_and_clean(
        prob,
        threshold=config.postprocess.confidence_threshold,
        morphology_kernel=config.postprocess.morphology_kernel,
        min_pixels=min_pixels,
    )
    features = mask_to_features(
        mask,
        transform=pair.before.transform,
        crs=pair.before.crs,
        confidence=prob,
    )

    result = ChangeDetectionResult(
        probability=prob,
        mask=mask,
        features=features,
        transform=pair.before.transform,
        crs=pair.before.crs,
        config=config,
    )

    if output_dir is not None:
        result.write_outputs(output_dir)
    return result
