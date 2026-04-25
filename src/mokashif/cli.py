"""Typer CLI for Mokashif.

Examples
--------
Detect change in two GeoTIFFs::

    mokashif detect --before a.tif --after b.tif --output ./out

Pre-fetch model weights so subsequent runs are fully offline::

    mokashif download-weights --target ./models_cache

Show package metadata::

    mokashif info
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from mokashif import __version__
from mokashif.config import (
    ModelConfig,
    PipelineConfig,
    PostprocessConfig,
    PreprocessConfig,
    TilingConfig,
)
from mokashif.pipeline.api import detect_changes

app = typer.Typer(
    add_completion=False,
    help="Mokashif (مكشاف) - state-of-the-art zero-shot satellite change detection.",
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, markup=True)],
        force=True,
    )


@app.callback()
def _main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    _setup_logging(verbose)


@app.command()
def detect(
    before: Path = typer.Option(..., "--before", "-b", exists=True, help="Earlier image."),
    after: Path = typer.Option(..., "--after", "-a", exists=True, help="Later image."),
    output: Path = typer.Option(Path("./results"), "--output", "-o", help="Output dir."),
    encoder: str = typer.Option("dinov2_base", "--encoder", "-e"),
    device: str = typer.Option("auto", "--device", "-d", help="cpu | cuda | mps | auto"),
    tile_size: int = typer.Option(512, "--tile-size"),
    overlap: int = typer.Option(64, "--overlap"),
    threshold: float = typer.Option(0.5, "--confidence-threshold", "-t", min=0.0, max=1.0),
    min_area_m2: float = typer.Option(100.0, "--min-area-m2"),
    coregister: bool = typer.Option(True, "--coregister/--no-coregister"),
    weights_dir: Path = typer.Option(Path("./models_cache"), "--weights-dir"),
) -> None:
    """Run change detection on a pair of rasters."""
    cfg = PipelineConfig(
        preprocess=PreprocessConfig(coregister=coregister),
        tiling=TilingConfig(tile_size=tile_size, overlap=overlap),
        model=ModelConfig(encoder=encoder, weights_dir=weights_dir),
        postprocess=PostprocessConfig(
            confidence_threshold=threshold,
            min_object_area_m2=min_area_m2,
        ),
        device=device,  # type: ignore[arg-type]
    )

    console.rule("[bold cyan]Mokashif[/]  -  change detection")
    result = detect_changes(before, after, config=cfg, device=device, output_dir=output)
    console.print(f"[green]{result.summary()}[/]")
    console.print(f"Outputs written to: [bold]{output}[/]")


@app.command("download-weights")
def download_weights(
    target: Path = typer.Option(Path("./models_cache"), "--target"),
    encoder: str = typer.Option("dinov2_base", "--encoder"),
) -> None:
    """Pre-fetch model weights for fully offline operation."""
    target.mkdir(parents=True, exist_ok=True)
    console.print(f"Downloading weights for [bold]{encoder}[/] -> {target} ...")
    from mokashif.models.encoders import build_encoder

    build_encoder(encoder, weights_dir=target)
    console.print("[green]Done.[/] Set MOKASHIF_OFFLINE=1 to enforce offline mode.")


@app.command()
def info() -> None:
    """Show package version + runtime info."""
    import platform
    import sys

    import torch

    console.print(f"[bold cyan]mokashif[/] {__version__}")
    console.print(f"Python  : {sys.version.split()[0]}")
    console.print(f"Platform: {platform.platform()}")
    console.print(f"PyTorch : {torch.__version__}")
    console.print(f"CUDA    : {'yes' if torch.cuda.is_available() else 'no'}")
    if torch.cuda.is_available():
        console.print(f"  device: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":  # pragma: no cover
    app()
