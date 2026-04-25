"""Optional offline FastAPI server.

Run with ``uvicorn mokashif.api:app --host 127.0.0.1 --port 8000``.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from mokashif import __version__
from mokashif.pipeline.api import detect_changes

app = FastAPI(title="Mokashif", version=__version__, docs_url="/docs")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/detect")
async def detect(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    threshold: float = 0.5,
) -> JSONResponse:
    """Detect changes between two uploaded rasters and return GeoJSON."""
    with NamedTemporaryFile(
        suffix=Path(before.filename or "before.tif").suffix, delete=False
    ) as fb:
        fb.write(await before.read())
        before_path = Path(fb.name)
    with NamedTemporaryFile(suffix=Path(after.filename or "after.tif").suffix, delete=False) as fa:
        fa.write(await after.read())
        after_path = Path(fa.name)

    from mokashif.config import PipelineConfig, PostprocessConfig

    config = PipelineConfig(
        postprocess=PostprocessConfig(confidence_threshold=threshold),
    )
    result = detect_changes(before_path, after_path, config=config)
    return JSONResponse(
        {
            "summary": result.summary(),
            "features": result.features,
        }
    )
