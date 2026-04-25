"""End-to-end pipeline tests on synthetic geotiffs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")
from mokashif.config import ModelConfig, PipelineConfig, PreprocessConfig, TilingConfig
from mokashif.pipeline.api import detect_changes


def _config_for_tests() -> PipelineConfig:
    return PipelineConfig(
        preprocess=PreprocessConfig(coregister=False, normalize="percentile"),
        tiling=TilingConfig(tile_size=128, overlap=16),
        model=ModelConfig(encoder="identity"),  # type: ignore[arg-type]
        device="cpu",
    )


def test_pipeline_runs_on_synthetic_pair(synthetic_pair: tuple[Path, Path]) -> None:
    before, after = synthetic_pair
    cfg = _config_for_tests()
    result = detect_changes(before, after, config=cfg, device="cpu")

    assert result.probability.shape == (256, 256)
    assert result.mask.dtype == np.uint8
    # Should detect *something*.
    assert result.mask.sum() > 0
    # The injected change is at rows/cols 96:160.
    inside = result.mask[96:160, 96:160].mean()
    outside_mask = np.ones_like(result.mask, dtype=bool)
    outside_mask[96:160, 96:160] = False
    outside = result.mask[outside_mask].mean()
    assert inside > outside


def test_pipeline_writes_outputs(synthetic_pair: tuple[Path, Path], tmp_path: Path) -> None:
    before, after = synthetic_pair
    out_dir = tmp_path / "out"
    cfg = _config_for_tests()
    result = detect_changes(before, after, config=cfg, device="cpu", output_dir=out_dir)

    assert (out_dir / "confidence.tif").exists()
    assert (out_dir / "change_mask.tif").exists()
    assert (out_dir / "changes.geojson").exists()

    # Result summary should mention the polygon count.
    text = result.summary()
    assert "polygon" in text


def test_pipeline_summary_shape(synthetic_pair: tuple[Path, Path]) -> None:
    before, after = synthetic_pair
    result = detect_changes(before, after, config=_config_for_tests(), device="cpu")
    s = result.summary()
    assert "Mokashif" in s
