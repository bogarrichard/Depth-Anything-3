#!/usr/bin/env python
"""Minimal standalone Depth Anything 3 demo.

Runs depth + camera pose estimation on a couple of sample images and saves
a colorized depth map next to each input, with no Jupyter/browser needed --
meant for a plain terminal (e.g. a fresh GitHub Codespace).

    python demo.py
    python demo.py --images path/to/a.jpg path/to/b.jpg --output-dir out/

For interactive use with more export formats (glb, 3DGS, etc.), see the
`da3` CLI instead (`da3 --help`).
"""

import argparse
from pathlib import Path
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

DEFAULT_IMAGES = [
    "assets/examples/SOH/000.png",
    "assets/examples/SOH/010.png",
]
# 0.35B params -- fast enough for a CPU demo. The giant/nested models are
# much better but noticeably slower without a GPU.
DEFAULT_MODEL = "depth-anything/DA3-LARGE-1.1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGES, help="Input image paths.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or local/HF path.")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use."
    )
    parser.add_argument("--process-res", type=int, default=504, help="Processing resolution.")
    parser.add_argument(
        "--output-dir", default="demo_output", help="Where to save input/depth PNGs."
    )
    args = parser.parse_args()

    print(f"Loading {args.model} on {args.device}...")
    model = DepthAnything3.from_pretrained(args.model).to(args.device)
    model.eval()

    print(f"Running inference on {len(args.images)} image(s)...")
    prediction = model.inference(
        image=args.images,
        process_res=args.process_res,
        process_res_method="upper_bound_resize",
    )
    print(f"Depth shape: {prediction.depth.shape}")
    print(
        f"Extrinsics: {'None' if prediction.extrinsics is None else prediction.extrinsics.shape}"
    )
    print(
        f"Intrinsics: {'None' if prediction.intrinsics is None else prediction.intrinsics.shape}"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(prediction.depth.shape[0]):
        if prediction.processed_images is not None:
            Image.fromarray(prediction.processed_images[i]).save(out_dir / f"input_{i:02d}.png")
        depth_vis = visualize_depth(prediction.depth[i], cmap="Spectral")
        Image.fromarray(depth_vis).save(out_dir / f"depth_{i:02d}.png")

    print(f"Saved input/depth PNGs to {out_dir}/")


if __name__ == "__main__":
    main()
