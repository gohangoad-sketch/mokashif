"""CLI smoke tests via Typer's runner."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("rasterio")
from mokashif.cli import app

runner = CliRunner()


def test_cli_info() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "mokashif" in result.stdout.lower()


def test_cli_detect_smoke(synthetic_pair: tuple[Path, Path], tmp_path: Path) -> None:
    before, after = synthetic_pair
    out_dir = tmp_path / "results"
    result = runner.invoke(
        app,
        [
            "detect",
            "--before",
            str(before),
            "--after",
            str(after),
            "--output",
            str(out_dir),
            "--encoder",
            "identity",
            "--device",
            "cpu",
            "--tile-size",
            "128",
            "--overlap",
            "16",
            "--no-coregister",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "changes.geojson").exists()
