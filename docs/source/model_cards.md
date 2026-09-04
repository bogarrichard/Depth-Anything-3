# Model Cards

Three series of models are released, each tailored for specific use cases in
visual geometry.

- **DA3 Main Series** (`DA3-Giant`, `DA3-Large`, `DA3-Base`, `DA3-Small`) --
  the flagship foundation models, trained with a unified depth-ray
  representation. By varying the input configuration, a single model can
  perform:
  - **Monocular depth estimation** -- a depth map from a single RGB image.
  - **Multi-view depth estimation** -- consistent depth maps from multiple
    images for high-quality fusion.
  - **Pose-conditioned depth estimation** -- superior depth consistency when
    camera poses are provided as input.
  - **Camera pose estimation** -- extrinsics and intrinsics from one or more
    images.
  - **3D Gaussian estimation** -- 3D Gaussians directly, for high-fidelity
    novel view synthesis.

- **DA3 Metric Series** (`DA3Metric-Large`) -- fine-tuned for metric depth
  estimation in monocular settings, for applications requiring real-world
  scale.

- **DA3 Monocular Series** (`DA3Mono-Large`) -- dedicated high-quality
  relative monocular depth estimation. Unlike disparity-based models (e.g.
  Depth Anything 2), it directly predicts depth for superior geometric
  accuracy.

The **Nested series** (`DA3Nested-Giant-Large`) combines an any-view giant
model with a metric model to reconstruct visual geometry at real-world
metric scale. `DA3-LARGE` generally achieves results comparable to VGGT.

```{note}
Models with the `-1.1` suffix were retrained after fixing a training bug --
prefer these refreshed checkpoints. The original `DA3NESTED-GIANT-LARGE`,
`DA3-GIANT`, and `DA3-LARGE` remain available but are deprecated; the
`-1.1` models perform noticeably better on street scenes.
```

## Available models

| Model Name | Params | Rel. Depth | Pose Est. | Pose Cond. | GS | Met. Depth | Sky Seg | License |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Nested** | | | | | | | | |
| [DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1) | 1.40B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CC BY-NC 4.0 |
| [DA3NESTED-GIANT-LARGE](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE) | 1.40B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CC BY-NC 4.0 |
| **Any-view Model** | | | | | | | | |
| [DA3-GIANT-1.1](https://huggingface.co/depth-anything/DA3-GIANT-1.1) | 1.15B | ✅ | ✅ | ✅ | ✅ | | | CC BY-NC 4.0 |
| [DA3-GIANT](https://huggingface.co/depth-anything/DA3-GIANT) | 1.15B | ✅ | ✅ | ✅ | ✅ | | | CC BY-NC 4.0 |
| [DA3-LARGE-1.1](https://huggingface.co/depth-anything/DA3-LARGE-1.1) | 0.35B | ✅ | ✅ | ✅ | | | | CC BY-NC 4.0 |
| [DA3-LARGE](https://huggingface.co/depth-anything/DA3-LARGE) | 0.35B | ✅ | ✅ | ✅ | | | | CC BY-NC 4.0 |
| [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) | 0.12B | ✅ | ✅ | ✅ | | | | Apache 2.0 |
| [DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL) | 0.08B | ✅ | ✅ | ✅ | | | | Apache 2.0 |
| **Monocular Metric Depth** | | | | | | | | |
| [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE) | 0.35B | ✅ | | | | ✅ | ✅ | Apache 2.0 |
| **Monocular Depth** | | | | | | | | |
| [DA3MONO-LARGE](https://huggingface.co/depth-anything/DA3MONO-LARGE) | 0.35B | ✅ | | | | | ✅ | Apache 2.0 |

## Choosing a model

- Starting out, or need the widest feature set (pose, depth, 3DGS, metric
  scale)? Use `DA3NESTED-GIANT-LARGE-1.1`.
- Want depth + pose without the Gaussian branch, at a fraction of the size?
  `DA3-LARGE-1.1` is the sweet spot (comparable to VGGT).
- Need metric (real-world scale) monocular depth specifically?
  `DA3METRIC-LARGE` -- see the conversion formula below.
- CPU or resource-constrained testing? `DA3-SMALL` (0.08B).

Load any of them by Hugging Face repo ID or a local directory -- see
{doc}`usage/python_api`.

## Metric depth conversion

To obtain metric depth in meters from `DA3METRIC-LARGE`:

```
metric_depth = focal * net_output / 300.
```

where `focal` is the focal length in pixels (typically the average of `fx`
and `fy` from the camera intrinsic matrix `K`). The output of
`DA3NESTED-GIANT-LARGE` is already in meters.

## Ray head vs. camera decoder (`use_ray_pose`)

The API and CLI both accept `use_ray_pose`: when set, the model derives
camera pose from a ray head instead of the camera decoder, which is
generally slightly slower but more accurate. Default is `False`, for faster
inference.

AUC3 results for `DA3NESTED-GIANT-LARGE`:

| Model | HiRoom | ETH3D | DTU | 7Scenes | ScanNet++ |
|---|---|---|---|---|---|
| `ray_head` | 84.4 | 52.6 | 93.9 | 29.5 | 89.4 |
| `cam_head` | 80.3 | 48.4 | 94.1 | 28.5 | 85.0 |
