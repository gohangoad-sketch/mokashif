"""Zero-shot change detectors built on top of an encoder."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

from mokashif.models.encoders import Encoder, _to_tensor


class ChangeDetector(ABC):
    """Abstract change detector: takes two arrays, returns a probability map."""

    @abstractmethod
    def detect(
        self,
        before: np.ndarray,
        after: np.ndarray,
        device: torch.device | str = "cpu",
    ) -> np.ndarray:
        """Return a (H, W) float32 array in [0, 1]."""


class SimilarityChangeDetector(ChangeDetector):
    """Encoder-features + dissimilarity = change.

    For each spatial location we compute a similarity (cosine by default)
    between the two feature maps and turn the dissimilarity into a probability
    via a logistic squash. The result is up-sampled back to the input
    resolution.

    This is intentionally simple but performs *very* well in practice when the
    encoder is strong (DINOv2) - it is the engine inside several recent
    zero-shot change-detection papers (e.g. AnyChange).
    """

    def __init__(
        self,
        encoder: Encoder,
        similarity: str = "cosine",
        temperature: float = 8.0,
    ) -> None:
        self.encoder = encoder
        self.similarity = similarity
        self.temperature = float(temperature)

    @torch.inference_mode()
    def detect(
        self,
        before: np.ndarray,
        after: np.ndarray,
        device: torch.device | str = "cpu",
    ) -> np.ndarray:
        device = torch.device(device)
        self.encoder.to(device).eval()

        b = _to_tensor(before, device)
        a = _to_tensor(after, device)
        target_size = b.shape[-2:]

        f_b = self.encoder(b)
        f_a = self.encoder(a)

        if f_b.shape != f_a.shape:
            f_a = F.interpolate(f_a, size=f_b.shape[-2:], mode="bilinear", align_corners=False)

        if self.similarity == "cosine":
            f_b_n = F.normalize(f_b, dim=1)
            f_a_n = F.normalize(f_a, dim=1)
            sim = (f_b_n * f_a_n).sum(dim=1)  # (B, H', W')
            diss = 1.0 - sim  # in [0, 2]
        elif self.similarity == "l2":
            diss = (f_b - f_a).pow(2).mean(dim=1).sqrt()
            diss = diss / diss.amax().clamp(min=1e-6)
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")

        # Squash to a probability with a learned-style logistic on the
        # dissimilarity. The temperature controls how sharp the boundary is.
        prob = torch.sigmoid(self.temperature * (diss - diss.median()))

        # Up-sample to original resolution.
        prob = F.interpolate(
            prob.unsqueeze(1),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        out = prob[0].detach().cpu().numpy().astype(np.float32)
        return out
