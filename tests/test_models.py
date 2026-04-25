"""Tests for encoders + change detector (using the lightweight identity encoder)."""

from __future__ import annotations

import numpy as np
import torch

from mokashif.models.change_detector import SimilarityChangeDetector
from mokashif.models.encoders import IdentityEncoder, build_encoder


def test_build_encoder_identity() -> None:
    enc = build_encoder("identity")
    assert isinstance(enc, IdentityEncoder)


def test_identity_encoder_shape() -> None:
    enc = IdentityEncoder(stride=8)
    x = torch.zeros(1, 3, 64, 64)
    y = enc(x)
    assert y.shape == (1, 3, 8, 8)


def test_similarity_detects_injected_change() -> None:
    rng = np.random.default_rng(7)
    h = w = 96
    before = rng.random((3, h, w), dtype=np.float32)
    after = before.copy()
    # Inject change in known region.
    after[:, 32:64, 32:64] = 1.0

    enc = IdentityEncoder(stride=8)
    det = SimilarityChangeDetector(enc, similarity="cosine")
    prob = det.detect(before, after, device="cpu")
    assert prob.shape == (h, w)
    assert prob.min() >= 0.0
    assert prob.max() <= 1.0
    # Mean probability inside the changed region should beat outside.
    inside = prob[32:64, 32:64].mean()
    outside_mask = np.ones_like(prob, dtype=bool)
    outside_mask[32:64, 32:64] = False
    outside = prob[outside_mask].mean()
    assert inside > outside, f"inside={inside:.3f} not > outside={outside:.3f}"
