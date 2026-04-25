"""Mokashif - state-of-the-art zero-shot satellite change detection."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mokashif")
except PackageNotFoundError:  # pragma: no cover - editable install fallback
    __version__ = "0.0.0+local"

from mokashif.pipeline.api import ChangeDetectionResult, detect_changes

__all__ = [
    "ChangeDetectionResult",
    "__version__",
    "detect_changes",
]
