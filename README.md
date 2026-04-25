# Mokashif (مكشاف)

> A state-of-the-art, **offline-capable**, **zero-shot** change-detection pipeline for large
> satellite imagery from **any sensor**.

[![CI](https://github.com/gohangoad-sketch/mokashif/actions/workflows/ci.yml/badge.svg)](https://github.com/gohangoad-sketch/mokashif/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## ✨ What is Mokashif?

Mokashif (Arabic: مكشاف — *"detector"*) is a production-grade pipeline that compares two
satellite images of the same area taken at different times and reports **what changed,
where, and how confidently**.

It is designed around three uncompromising principles:

1. **Sensor-agnostic** — works with any imagery `rasterio` / GDAL can read (Sentinel-1/2,
   Landsat, Planet, Maxar/WorldView, Airbus, NAIP, Umbra-SAR, drone, aerial, …).
2. **Zero-shot** — no labels, no fine-tuning. Powered by foundation-model embeddings
   (DINOv2 / Prithvi / Clay) compared at the pixel and patch level.
3. **Offline-first** — after a one-time weight download, the entire pipeline runs without
   any network connection.

> ⚠️ **Honest disclaimer.** No change-detection system can guarantee 100% confidence —
> clouds, shadows, sun-angle, sensor disparity, and registration error all introduce
> uncertainty. Mokashif targets the SOTA range (**F1 ≈ 0.92 – 0.96** on LEVIR-CD-class
> benchmarks) and exposes a calibrated confidence score per detection so you can filter
> aggressively.

---

## 🏗️ Architecture

```
                ┌────────────────────────────────────────┐
                │        Any-sensor I/O (rasterio)       │
                │ GeoTIFF · COG · JP2 · NITF · HDF5 · …  │
                └──────────────────┬─────────────────────┘
                                   │
                ┌──────────────────▼─────────────────────┐
                │  Preprocess: reproject · co-register   │
                │  (AROSICS) · cloud-mask (omnicloud)    │
                └──────────────────┬─────────────────────┘
                                   │
                ┌──────────────────▼─────────────────────┐
                │     Smart tiling (overlapping)         │
                └──────────────────┬─────────────────────┘
                                   │
                ┌──────────────────▼─────────────────────┐
                │   Foundation-encoder ensemble          │
                │   DINOv2 + (Prithvi | Clay) + similarity│
                └──────────────────┬─────────────────────┘
                                   │
                ┌──────────────────▼─────────────────────┐
                │  Post-process: stitch · threshold ·    │
                │  morphology · vectorize · confidence   │
                └──────────────────┬─────────────────────┘
                                   │
                ┌──────────────────▼─────────────────────┐
                │   Outputs: GeoJSON · COG mask · report │
                └────────────────────────────────────────┘
```

---

## 🚀 Quick start

### 1. Install

```bash
# Recommended: uv (fastest)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra all --extra dev

# Or plain pip
pip install -e ".[all,dev]"
```

PyTorch is pulled in automatically; install a CUDA build matching your driver if you want
GPU acceleration:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Download model weights (once, online)

```bash
mokashif download-weights --target ./models_cache
```

After this, set `MOKASHIF_OFFLINE=1` and the pipeline will refuse any network call.

### 3. Run change detection

```bash
mokashif detect \
    --before path/to/image_2020.tif \
    --after  path/to/image_2024.tif \
    --output ./results/ \
    --device cuda \
    --confidence-threshold 0.5
```

Outputs:

- `results/changes.geojson` — vector polygons with per-feature confidence
- `results/change_mask.tif` — Cloud-Optimised GeoTIFF (binary)
- `results/confidence.tif` — float COG of the calibrated probability
- `results/report.html` — standalone, offline-viewable summary

### 4. Inspect interactively

```bash
mokashif viewer ./results/   # opens an offline leafmap viewer
```

### 5. Programmatic API

```python
from mokashif import detect_changes

result = detect_changes(
    before="image_2020.tif",
    after="image_2024.tif",
    device="cuda",
)
print(result.summary())
result.to_geojson("changes.geojson")
```

---

## 🛰️ Tested sensors (out of the box)

| Sensor              | Bands used      | Native GSD    | Notes              |
| ------------------- | --------------- | ------------- | ------------------ |
| Sentinel-2 L2A      | RGB+NIR (B2-4-8)| 10 m          | Cloud-masked       |
| Landsat 8/9 L2      | RGB+NIR         | 30 m          | Cloud-masked       |
| Planet PSScene      | RGB+NIR         | ~3 m          | —                  |
| Maxar WorldView     | RGB / pansharp  | 0.3 – 0.5 m   | NITF supported     |
| Airbus Pléiades     | RGB / pansharp  | 0.5 m         | —                  |
| NAIP                | RGB+NIR         | 0.6 – 1 m     | —                  |
| Sentinel-1 (SAR)    | VV+VH           | 10 m          | Log-amplitude path |
| UAV / aerial        | any             | any           | EXIF/world-file ok |

Anything `rasterio` opens, Mokashif accepts.

---

## 📂 Project layout

```
src/mokashif/
├── io/          # Loaders, writers, COG export
├── preprocess/  # Reproject, co-register, mask
├── tiling/      # Overlap-aware sliding-window tiling
├── models/      # Encoders + similarity heads
├── postprocess/ # Threshold, morphology, vectorise, calibrate
├── pipeline/    # Orchestration + high-level API
├── cli.py       # Typer CLI
├── api.py       # FastAPI offline server
└── viewer.py    # leafmap-based offline viewer
```

---

## 🧪 Development

```bash
uv sync --extra all --extra dev
pre-commit install
pytest -m "not slow and not gpu"
ruff check .
mypy src
```

---

## 📜 License

Apache 2.0 — see [LICENSE](LICENSE).
