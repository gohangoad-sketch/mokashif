"""Pre-fetch model weights so the pipeline can run fully offline.

Usage::

    python scripts/download_weights.py --target ./models_cache --encoder dinov2_base
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Mokashif model weights.")
    parser.add_argument("--target", type=Path, default=Path("./models_cache"))
    parser.add_argument("--encoder", default="dinov2_base")
    args = parser.parse_args()

    args.target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.encoder} weights into {args.target} ...")
    from mokashif.models.encoders import build_encoder

    enc = build_encoder(args.encoder, weights_dir=args.target)
    print(f"Done. Encoder feature_dim={enc.feature_dim}, stride={enc.stride}.")


if __name__ == "__main__":
    main()
