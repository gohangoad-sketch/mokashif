"""Centralised configuration objects for the Mokashif pipeline.

We intentionally use a small number of explicit dataclasses (instead of one giant
``Settings`` blob) so individual stages can be configured and reused independently.
Pydantic gives us validation + nice error messages without coupling us to any
particular serialiser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DeviceLiteral = Literal["cpu", "cuda", "mps", "auto"]


class IOConfig(BaseModel):
    """How we read / write rasters."""

    target_resolution_m: float | None = Field(
        default=None,
        description="If set, both inputs are resampled to this GSD (metres) before processing.",
    )
    target_crs: str | None = Field(
        default=None,
        description="Override target CRS (EPSG:xxxx). Defaults to the CRS of the *before* image.",
    )
    resampling: Literal["bilinear", "cubic", "nearest", "average"] = "bilinear"


class PreprocessConfig(BaseModel):
    """Co-registration, masking and normalisation."""

    coregister: bool = Field(
        default=True,
        description="Run AROSICS sub-pixel co-registration if available.",
    )
    cloud_mask: bool = Field(
        default=False,
        description="Run omnicloudmask on each input. Requires extra dependency + Sentinel/Landsat-like bands.",
    )
    normalize: Literal["percentile", "zscore", "none"] = "percentile"
    percentile_low: float = 2.0
    percentile_high: float = 98.0


class TilingConfig(BaseModel):
    """Sliding-window tile generation."""

    tile_size: int = Field(default=512, ge=64, le=2048)
    overlap: int = Field(default=64, ge=0)
    pad_value: float = 0.0

    @field_validator("overlap")
    @classmethod
    def _overlap_below_tile(cls, v: int, info) -> int:
        ts = info.data.get("tile_size", 512)
        if v >= ts:
            raise ValueError(f"overlap ({v}) must be smaller than tile_size ({ts})")
        return v


class ModelConfig(BaseModel):
    """Which encoder(s) to ensemble."""

    encoder: str = Field(
        default="dinov2_base",
        description=(
            "Encoder name. Built-in: 'identity' (test/baseline), 'imagenet_resnet50', "
            "'dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant'."
        ),
    )
    use_prithvi: bool = Field(
        default=False,
        description="Add NASA/IBM Prithvi-EO-2.0 to the ensemble (extras: foundation).",
    )
    similarity: Literal["cosine", "l2", "mahalanobis"] = "cosine"
    weights_dir: Path = Field(default=Path("./models_cache"))


class PostprocessConfig(BaseModel):
    """How to turn similarity maps into clean polygons."""

    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_object_area_m2: float = Field(
        default=100.0,
        description="Drop any change polygon smaller than this area (m^2).",
    )
    morphology_kernel: int = Field(default=3, ge=1, le=15)
    calibrate: bool = Field(
        default=True,
        description="Histogram-based confidence calibration so 0.5 ~ true 50% precision.",
    )


class PipelineConfig(BaseModel):
    """Full pipeline configuration. Compose the sub-configs."""

    io: IOConfig = Field(default_factory=IOConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    device: DeviceLiteral = "auto"
    batch_size: int = Field(default=8, ge=1)
    num_workers: int = Field(default=4, ge=0)
    offline: bool = Field(
        default=False,
        description="If True, raise immediately on any network access attempt.",
    )

    model_config = {"protected_namespaces": ()}
