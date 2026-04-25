"""Overlap-aware tiling.

For 100x100 km imagery at 1 m/px we may have 100,000^2 pixels => billions of
samples. We process the scene in overlapping ``tile_size`` windows and blend
the resulting probability maps using a Hann-window weighting so seams
disappear.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    """A tile read from a larger image.

    Attributes
    ----------
    row, col
        Top-left pixel coordinate in the parent image.
    height, width
        Tile spatial dimensions (post-padding, may equal ``tile_size``).
    valid_height, valid_width
        Number of pixels copied from the parent image (the rest is padding).
    """

    row: int
    col: int
    height: int
    width: int
    valid_height: int
    valid_width: int


def generate_tiles(
    height: int,
    width: int,
    tile_size: int = 512,
    overlap: int = 64,
) -> list[Tile]:
    """Compute the set of tile windows covering an image.

    Tiles overlap by ``overlap`` pixels and the right/bottom tiles are
    snapped so they always end at the image border (no truncation).
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")

    stride = tile_size - overlap
    tiles: list[Tile] = []
    for r in range(0, max(height - overlap, 1), stride):
        # Snap last tile so it ends at image bottom.
        if r + tile_size > height:
            r = max(height - tile_size, 0)
        for c in range(0, max(width - overlap, 1), stride):
            if c + tile_size > width:
                c = max(width - tile_size, 0)
            vh = min(tile_size, height - r)
            vw = min(tile_size, width - c)
            tiles.append(
                Tile(
                    row=r,
                    col=c,
                    height=tile_size,
                    width=tile_size,
                    valid_height=vh,
                    valid_width=vw,
                )
            )
            if c + tile_size >= width:
                break
        if r + tile_size >= height:
            break

    # De-duplicate (when image smaller than tile, snapping can produce dupes).
    seen: set[tuple[int, int]] = set()
    unique: list[Tile] = []
    for t in tiles:
        key = (t.row, t.col)
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique


def extract_tile(image: np.ndarray, tile: Tile, pad_value: float = 0.0) -> np.ndarray:
    """Extract a tile from ``image`` (C, H, W), padding right/bottom if needed."""
    if image.ndim != 3:
        raise ValueError(f"Expected (C, H, W) image, got shape {image.shape}")
    c = image.shape[0]
    out = np.full((c, tile.height, tile.width), pad_value, dtype=image.dtype)
    out[:, : tile.valid_height, : tile.valid_width] = image[
        :,
        tile.row : tile.row + tile.valid_height,
        tile.col : tile.col + tile.valid_width,
    ]
    return out


def _hann_window(size: int) -> np.ndarray:
    """1-D Hann window without the zero endpoints (so weights stay > 0)."""
    n = np.arange(size)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * (n + 1) / (size + 1))
    return w.astype(np.float32)


def stitch_tiles(
    height: int,
    width: int,
    tiles: list[Tile],
    tile_outputs: list[np.ndarray],
) -> np.ndarray:
    """Blend tile probability maps back into a single (H, W) array.

    Each ``tile_outputs[i]`` is a (tile.height, tile.width) float array.
    """
    if len(tiles) != len(tile_outputs):
        raise ValueError("tiles and tile_outputs must have the same length")
    if not tiles:
        return np.zeros((height, width), dtype=np.float32)

    canvas = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    th, tw = tiles[0].height, tiles[0].width
    win = np.outer(_hann_window(th), _hann_window(tw)).astype(np.float32)

    for tile, out in zip(tiles, tile_outputs, strict=True):
        vh, vw = tile.valid_height, tile.valid_width
        sub = out[:vh, :vw]
        w_sub = win[:vh, :vw]
        canvas[
            tile.row : tile.row + vh,
            tile.col : tile.col + vw,
        ] += sub * w_sub
        weight[
            tile.row : tile.row + vh,
            tile.col : tile.col + vw,
        ] += w_sub

    canvas /= np.maximum(weight, 1e-6)
    return canvas


def iterate_tile_pairs(
    before: np.ndarray,
    after: np.ndarray,
    tiles: list[Tile],
    pad_value: float = 0.0,
) -> Iterator[tuple[Tile, np.ndarray, np.ndarray]]:
    """Yield (tile, before_patch, after_patch) for each tile."""
    for t in tiles:
        yield t, extract_tile(before, t, pad_value), extract_tile(after, t, pad_value)
