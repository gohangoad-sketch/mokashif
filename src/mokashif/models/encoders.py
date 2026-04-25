"""Backbone encoders used by the change detector.

We keep a tiny abstract :class:`Encoder` interface and a few concrete
implementations:

- ``DinoV2Encoder``  : Meta DINOv2 ViT-S/B/L via torch.hub or transformers.
- ``ResNet50Encoder``: ImageNet ResNet-50 via torchvision (lightweight fallback,
  also used in CI where downloading DINOv2 is too expensive).
- ``IdentityEncoder``: deterministic feature = downsampled image. Used in unit
  tests so the pipeline stays runnable without any network or large weights.

All encoders return spatial features of shape ``(B, D, H', W')`` where
``H' = H / stride`` and ``W' = W / stride``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class Encoder(ABC, nn.Module):
    """Abstract spatial-feature encoder."""

    stride: int = 1
    feature_dim: int = 0

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (B?, C, H, W) numpy float32 in [0,1] to a tensor on device."""
    if arr.ndim == 3:
        arr = arr[None]
    t = torch.as_tensor(arr, dtype=torch.float32, device=device)
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    elif t.shape[1] > 3:
        t = t[:, :3]
    return t


class IdentityEncoder(Encoder):
    """Down-sample the image to a feature map. Deterministic, no training.

    Useful for tests and as a *very* fast baseline. Surprisingly competitive on
    coarse change because we still do per-channel cosine similarity.
    """

    def __init__(self, stride: int = 8) -> None:
        super().__init__()
        self.stride = stride
        self.feature_dim = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)


class ResNet50Encoder(Encoder):
    """ImageNet-pretrained ResNet-50 truncated at the layer-3 feature map."""

    def __init__(self, weights_path: str | Path | None = None) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet50
        except ImportError as e:  # pragma: no cover
            raise ImportError("torchvision required for ResNet50Encoder") from e

        weights = None
        if weights_path is None:
            try:
                from torchvision.models import ResNet50_Weights

                weights = ResNet50_Weights.IMAGENET1K_V2
            except Exception:
                weights = None

        net = resnet50(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.stride = 16
        self.feature_dim = 1024

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean")
        std = self.get_buffer("std")
        x = (x - mean) / std
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class DinoV2Encoder(Encoder):
    """Meta DINOv2 ViT encoder.

    Loads via ``torch.hub`` with ``force_reload=False`` so cached weights are
    used offline. If network access is forbidden and the cache is empty this
    falls back to :class:`ResNet50Encoder` with a warning.
    """

    SIZE_TO_DIM: ClassVar[dict[str, int]] = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }

    def __init__(self, size: str = "base", weights_dir: Path | None = None) -> None:
        super().__init__()
        self.stride = 14  # DINOv2 patch size
        if size not in self.SIZE_TO_DIM:
            raise ValueError(f"Unknown DINOv2 size: {size}")
        self.feature_dim = self.SIZE_TO_DIM[size]
        if weights_dir is not None:
            torch.hub.set_dir(str(weights_dir))

        try:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                f"dinov2_vit{size[0]}14",
                pretrained=True,
                trust_repo=True,
            )
        except Exception as e:
            logger.warning("Failed to load DINOv2 (%s); falling back to ResNet50 features.", e)
            self.backbone = None
            self._fallback: ResNet50Encoder | None = ResNet50Encoder()
            self.stride = self._fallback.stride
            self.feature_dim = self._fallback.feature_dim
            return

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self._fallback = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            assert self._fallback is not None
            return self._fallback(x)

        mean = self.get_buffer("mean")
        std = self.get_buffer("std")
        x = (x - mean) / std
        # Pad to multiple of 14.
        h, w = x.shape[-2:]
        ph = (self.stride - h % self.stride) % self.stride
        pw = (self.stride - w % self.stride) % self.stride
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        feat_dict = self.backbone.forward_features(x)
        # forward_features returns dict with 'x_norm_patchtokens': (B, N, D)
        tokens = feat_dict["x_norm_patchtokens"]
        b = x.shape[0]
        hh = x.shape[-2] // self.stride
        ww = x.shape[-1] // self.stride
        return tokens.transpose(1, 2).reshape(b, self.feature_dim, hh, ww)


def build_encoder(name: str, weights_dir: Path | None = None) -> Encoder:
    """Factory used by configs / CLI."""
    name = name.lower()
    if name in {"identity", "test"}:
        return IdentityEncoder()
    if name in {"resnet50", "imagenet_resnet50"}:
        return ResNet50Encoder()
    if name.startswith("dinov2"):
        size = name.split("_")[-1] if "_" in name else "base"
        return DinoV2Encoder(size=size, weights_dir=weights_dir)
    raise ValueError(f"Unknown encoder: {name}")


def encode_image(
    encoder: Encoder,
    image: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Run an encoder on a numpy (C,H,W) or batch (B,C,H,W) array."""
    encoder.eval()
    encoder.to(device)
    with torch.inference_mode():
        x = _to_tensor(image, device)
        return encoder(x)
