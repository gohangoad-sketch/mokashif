"""Tests for the tiling utilities."""

from __future__ import annotations

import numpy as np
import pytest

from mokashif.tiling.window import (
    Tile,
    extract_tile,
    generate_tiles,
    iterate_tile_pairs,
    stitch_tiles,
)


def test_generate_tiles_covers_image() -> None:
    tiles = generate_tiles(height=1000, width=800, tile_size=256, overlap=32)
    # Every pixel should be inside at least one tile.
    coverage = np.zeros((1000, 800), dtype=bool)
    for t in tiles:
        coverage[t.row : t.row + t.valid_height, t.col : t.col + t.valid_width] = True
    assert coverage.all()


def test_generate_tiles_small_image() -> None:
    tiles = generate_tiles(height=100, width=100, tile_size=256, overlap=32)
    assert len(tiles) == 1
    t = tiles[0]
    assert t.valid_height == 100
    assert t.valid_width == 100


@pytest.mark.parametrize("overlap", [0, 16, 64])
def test_generate_tiles_overlap_is_respected(overlap: int) -> None:
    tiles = generate_tiles(height=512, width=512, tile_size=256, overlap=overlap)
    rows = sorted({t.row for t in tiles})
    expected_stride = 256 - overlap
    if len(rows) > 1:
        # Strides should be at most the configured stride (last tile may snap).
        diffs = np.diff(rows)
        assert (diffs <= expected_stride).all()


def test_extract_tile_pads_correctly() -> None:
    img = np.arange(3 * 10 * 10, dtype=np.float32).reshape(3, 10, 10)
    tile = Tile(row=8, col=8, height=4, width=4, valid_height=2, valid_width=2)
    patch = extract_tile(img, tile, pad_value=-1.0)
    assert patch.shape == (3, 4, 4)
    # Top-left 2x2 is the real data.
    np.testing.assert_array_equal(patch[:, :2, :2], img[:, 8:10, 8:10])
    # Padding region matches pad_value.
    assert (patch[:, 2:, :] == -1.0).all()
    assert (patch[:, :, 2:] == -1.0).all()


def test_stitch_tiles_reconstructs_uniform_field() -> None:
    h, w = 200, 300
    tiles = generate_tiles(h, w, tile_size=64, overlap=16)
    outputs = [np.full((t.height, t.width), 0.7, dtype=np.float32) for t in tiles]
    stitched = stitch_tiles(h, w, tiles, outputs)
    assert stitched.shape == (h, w)
    np.testing.assert_allclose(stitched, 0.7, atol=1e-5)


def test_iterate_tile_pairs_lengths() -> None:
    before = np.zeros((3, 100, 100), dtype=np.float32)
    after = np.ones_like(before)
    tiles = generate_tiles(100, 100, tile_size=64, overlap=8)
    items = list(iterate_tile_pairs(before, after, tiles))
    assert len(items) == len(tiles)
    for t, b, a in items:
        assert b.shape == (3, t.height, t.width)
        assert a.shape == (3, t.height, t.width)
