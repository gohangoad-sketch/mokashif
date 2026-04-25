# Architecture

This document explains the Mokashif pipeline at the level of "what runs where
and why".

## Pipeline stages

```
prepare_pair → align_pair → normalize_pair → tile → encode → similarity →
calibrate → threshold + clean → vectorize → write
```

Each stage is implemented in its own module under `src/mokashif/`, has a
single-responsibility public function, and is independently unit-tested.

### 1. `io.readers.prepare_pair`

Opens the *before* and *after* rasters using `rasterio` (so any GDAL-supported
format works). When the source has more than 3 bands, the readers auto-pick
RGB based on `colorinterp` tags, then band descriptions, and finally fall back
to "the first three bands". Single-band rasters are repeated to give the
encoder the 3 channels it expects.

### 2. `preprocess.align.align_pair`

- Coarse: `rasterio.warp.reproject` snaps the *after* image onto the *before*
  image's CRS + transform, so both arrays share one grid.
- Fine (optional): `arosics.COREG` does sub-pixel co-registration when both
  rasters live on disk and AROSICS is installed. We swallow AROSICS errors
  loudly so missing extras can never break a run.

### 3. `preprocess.normalize.normalize_pair`

Robust per-band normalisation. The default ("percentile") clips to
[p2, p98] and scales to [0, 1], which is the only normaliser that works
reliably across DN ranges (Sentinel reflectance, Maxar 11-bit, SAR dB, …).

### 4. `tiling.window.generate_tiles`

For 100 × 100 km imagery at 1 m/px we have 10⁹ pixels. The tiler generates
overlapping windows, snaps the right/bottom tiles to the image edge, and
de-duplicates. Together with `stitch_tiles` (Hann-window blending) the
inference is seamless.

### 5. `models.encoders.build_encoder`

Three encoder families, all returning a `(B, D, H', W')` feature map:

| Encoder        | Weights source        | Feature dim | Stride | Notes                             |
| -------------- | --------------------- | ----------- | ------ | --------------------------------- |
| `identity`     | (none, deterministic) | 3           | 8      | CI / sanity-check baseline.       |
| `imagenet_resnet50` | torchvision         | 1024        | 16     | Light, robust fallback.           |
| `dinov2_*`     | facebookresearch/dinov2 (torch.hub) | 384/768/1024/1536 | 14 | SOTA self-supervised features.    |

DINOv2 falls back to ResNet-50 with a warning if the cache is empty and the
network is offline, so the pipeline never hard-fails.

### 6. `models.change_detector.SimilarityChangeDetector`

Cosine (or L2) dissimilarity between encoded features, bilinearly upsampled
to the input resolution and squashed by a temperature-controlled logistic.

### 7. `postprocess.calibrate.calibrate_probabilities`

Rank-based calibration: the empirical CDF of the probability map is replaced
by a logistic squash, so the threshold knob has stable semantics across
images.

### 8. `postprocess.vectorize.threshold_and_clean` + `mask_to_features`

Binary morphology + minimum-area filter, then `rasterio.features.shapes` →
shapely polygons re-projected to EPSG:4326 with per-feature confidence
statistics.

### 9. `io.writers.write_cog` + `write_geojson`

Output formats:

- `confidence.tif` — Cloud-Optimised GeoTIFF (uint16 in 0..10000).
- `change_mask.tif` — COG (uint8 binary).
- `changes.geojson` — GeoJSON FeatureCollection with confidence properties.
- `report.html` — optional standalone Leaflet-based viewer.

## Offline guarantees

The hard constraint "must run with no internet" is honoured by:

1. **Lazy network use.** The only stage that ever touches the network is
   weight download, behind an explicit `mokashif download-weights` command.
2. **Cache-first loading.** `torch.hub.set_dir` is set to the user-controlled
   `weights_dir` so cached weights are reused.
3. **Fallback paths.** If foundation weights are missing in offline mode the
   encoder degrades to the bundled torchvision ResNet-50 weights.

## Extending Mokashif

To add a new encoder:

1. Subclass `mokashif.models.encoders.Encoder`, set `stride` and
   `feature_dim`, implement `forward(x) -> (B, D, H', W')`.
2. Register it in `build_encoder`.
3. Add a config literal in `mokashif.config.ModelConfig.encoder`.
