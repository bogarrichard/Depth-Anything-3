#!/usr/bin/env python
"""Reconstruct a scene from a directory of images with Depth Anything 3.

    python reconstruct.py path/to/images
    rerun path/to/images.rrd

Writes a single Rerun recording holding the point cloud, the 3D Gaussians and
the cameras under `world`, in the model's own frame (its reference camera) and
scale -- metres with the default model. A transform on `world` levels the scene.

The Gaussians need a model with a GS head (`DA3-GIANT`, `DA3NESTED-GIANT-LARGE`);
pass `--no-gaussians` for any other. They are DA3's own feed-forward splats, which
look right from near the input cameras and get hazier the further the eye strays.
"""

import argparse
from pathlib import Path
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from PIL import Image, ImageOps

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import as_homogeneous

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SH_C0 = 0.28209479177387814  # degree-0 spherical harmonic, turns the DC term into a colour


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image_dir", type=Path, help="Directory containing the input images.")
    parser.add_argument(
        "-o", "--output", type=Path, help="Default: <image_dir>.rrd, next to the image directory."
    )
    parser.add_argument(
        "--model",
        default="DA3NESTED-GIANT-LARGE",
        help="Model name, e.g. DA3-LARGE-1.1; looked up under depth-anything/ on Hugging Face. "
        "Anything containing a slash is taken as is. Needs a multi-view model (not METRIC/MONO).",
    )
    parser.add_argument("--process-res", type=int, default=504, help="Processing resolution.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth image. All of them share one forward pass, so raise this when "
        "the GPU runs out of memory.",
    )
    parser.add_argument(
        "--gaussians",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also log 3D Gaussians.",
    )
    parser.add_argument(
        "--points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log the point cloud.",
    )
    parser.add_argument(
        "--cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log the cameras with their images.",
    )
    parser.add_argument(
        "--conf-percentile",
        type=float,
        default=40.0,
        help="Drop this percentage of the least confident non-sky pixels.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Randomly thin the point cloud down to this many points (0 = keep all).",
    )
    parser.add_argument(
        "--max-splat-size",
        type=float,
        default=6.0,
        help="Drop Gaussians larger than this many times the pixel they were predicted from.",
    )
    args = parser.parse_args()

    images = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    images = images[:: args.stride]
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")
    output = args.output or args.image_dir.resolve().with_suffix(".rrd")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model if "/" in args.model else f"depth-anything/{args.model}"
    model = DepthAnything3.from_pretrained(model_id).to(device).eval()
    # Opened here rather than passed as paths: DA3's own loader ignores the EXIF
    # orientation, which hands the model every portrait phone photo lying on its side.
    upright = [ImageOps.exif_transpose(Image.open(p)) for p in images]
    prediction = model.inference(
        image=upright, infer_gs=args.gaussians, process_res=args.process_res
    )
    depth, conf = prediction.depth, prediction.conf  # (N, H, W)
    height, width = depth.shape[1:]
    cam_to_world = np.linalg.inv(as_homogeneous(prediction.extrinsics))  # (N, 4, 4)

    # Confidence is `exp(x) + 1`, so strictly above 1 -- except for sky, which the
    # nested model resets to exactly 1.
    keep = conf > 1.0
    keep &= conf >= np.percentile(conf[keep], args.conf_percentile)

    rr.init("da3_reconstruction")
    rr.save(output, default_blueprint=rrb.Blueprint(rrb.Spatial3DView(origin="/")))
    rr.log("/", rr.ViewCoordinates.RDF, static=True)

    # DA3's world frame is its reference camera, which is rarely held level. A camera's
    # horizontal axis usually is, though, so "down" is the direction most nearly
    # perpendicular to every camera's x axis -- and, to break ties, nearest their y axes.
    # Everything is logged untouched under `world`; this one transform stands it upright.
    right, below = cam_to_world[:, :3, 0], cam_to_world[:, :3, 1]
    down = np.linalg.eigh(right.T @ right - 0.1 * below.T @ below)[1][:, 0]
    down *= np.sign(down @ below.sum(axis=0))
    forward = np.cross(np.cross(down, cam_to_world[0, :3, 2]), down)
    forward /= np.linalg.norm(forward)
    level_from_world = np.stack([np.cross(down, forward), down, forward])
    rr.log("world", rr.Transform3D(mat3x3=level_from_world), static=True)

    for i, image in enumerate(images if args.cameras else []):
        camera = f"world/cameras/{image.stem}"
        rr.log(
            camera,
            rr.Transform3D(translation=cam_to_world[i, :3, 3], mat3x3=cam_to_world[i, :3, :3]),
            rr.Pinhole(image_from_camera=prediction.intrinsics[i], resolution=[width, height]),
            rr.Image(prediction.processed_images[i]),
            static=True,
        )

    if args.points:
        # Back-project the depth maps: pixel -> camera ray -> world point.
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        pixels = np.stack([u, v, np.ones_like(u)], axis=-1).astype(np.float32)  # (H, W, 3)
        rays = np.einsum("nij,hwj->nhwi", np.linalg.inv(prediction.intrinsics), pixels)
        points = (
            np.einsum("nij,nhwj->nhwi", cam_to_world[:, :3, :3], rays * depth[..., None])
            + cam_to_world[:, None, None, :3, 3]
        )
        points, colors = points[keep], prediction.processed_images[keep]
        if 0 < args.max_points < len(points):
            chosen = np.random.default_rng(0).choice(len(points), args.max_points, replace=False)
            points, colors = points[chosen], colors[chosen]
        rr.log("world/point_cloud", rr.Points3D(points, colors=colors), static=True)

    if args.gaussians:
        gaussians = prediction.gaussians  # one per pixel, in (N, H, W) order
        # The nested model rescales depth and poses to metres only after the GS head
        # has run, which leaves the Gaussians in the pre-scaling frame.
        scale = (prediction.scale_factor or 1.0) if prediction.is_metric else 1.0

        # DA3 fits its splats to be seen from near the input cameras. Seen from anywhere
        # else the oversized ones turn the scene into fog, so those are dropped. Size is
        # measured against the footprint of the pixel a splat was predicted from.
        footprint = depth / prediction.intrinsics[:, :1, :1]  # (N, H, W), metres per pixel
        size = gaussians.scales[0].amax(dim=-1).float().cpu().numpy().reshape(depth.shape) * scale
        keep_splats = keep & (size <= args.max_splat_size * footprint)
        keep_flat = torch.from_numpy(keep_splats.reshape(-1)).to(gaussians.means.device)

        def kept(tensor: torch.Tensor) -> np.ndarray:
            return tensor[0][keep_flat].float().cpu().numpy()

        rgb = np.clip(0.5 + SH_C0 * kept(gaussians.harmonics[..., 0]), 0, 1)
        rgba = np.concatenate([rgb, kept(gaussians.opacities)[:, None]], axis=1)
        rr.log(
            "world/gaussians",
            rr.GaussianSplats3D(
                centers=kept(gaussians.means) * scale,
                scales=kept(gaussians.scales) * scale,
                quaternions=kept(gaussians.rotations)[:, [1, 2, 3, 0]],  # wxyz -> xyzw
                # As bytes: Rerun would take floats for linear colour and gamma-encode them.
                colors=(rgba * 255).round().astype(np.uint8),
            ),
            static=True,
        )

    print(f"Reconstruction of {len(images)} image(s) written to {output}")


if __name__ == "__main__":
    main()
