# Python API

Generated from the source docstrings -- always in sync with the installed
version. Runnable, task-oriented examples follow in {ref}`examples-python-api`
below.

## `DepthAnything3`

```{eval-rst}
.. autoclass:: depth_anything_3.api.DepthAnything3
   :members:
   :undoc-members:
   :show-inheritance:
```

## Return value: `Prediction`

```{eval-rst}
.. autoclass:: depth_anything_3.specs.Prediction

.. autoclass:: depth_anything_3.specs.Gaussians
```

(export-formats)=

## Export formats

Every export format is one function below, selected by name in
`export_format` (combine several with `-`, e.g. `"mini_npz-glb"`) and
dispatched by `export()`. Format-specific parameters (`export_kwargs`) are
that function's own keyword arguments.

```{eval-rst}
.. autofunction:: depth_anything_3.utils.export.export

.. autofunction:: depth_anything_3.utils.export.npz.export_to_npz

.. autofunction:: depth_anything_3.utils.export.npz.export_to_mini_npz

.. autofunction:: depth_anything_3.utils.export.glb.export_to_glb

.. autofunction:: depth_anything_3.utils.export.gs.export_to_gs_ply

.. autofunction:: depth_anything_3.utils.export.gs.export_to_gs_video

.. autofunction:: depth_anything_3.utils.export.feat_vis.export_to_feat_vis

.. autofunction:: depth_anything_3.utils.export.depth_vis.export_to_depth_vis
```

```{note}
`--export-format colmap` (`export_to_colmap`) needs the `colmap` extra
(`pycolmap`) and is left out of this auto-generated reference for that
reason -- see its docstring directly in
[`utils/export/colmap.py`](https://github.com/bogarrichard/Depth-Anything-3/blob/dev/src/depth_anything_3/utils/export/colmap.py)
on GitHub.
```

## Config system

```{eval-rst}
.. autofunction:: depth_anything_3.cfg.load_config

.. autofunction:: depth_anything_3.cfg.create_object
```

(examples-python-api)=

## Examples

### Basic depth estimation

```python
from depth_anything_3.api import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
model = model.to("cuda")

prediction = model.inference(["image1.jpg", "image2.jpg"])

print(prediction.depth.shape)             # (N, H, W)          float32
print(prediction.conf.shape)              # (N, H, W)          float32
print(prediction.extrinsics.shape)        # (N, 3, 4)          float32, opencv w2c
print(prediction.intrinsics.shape)        # (N, 3, 3)          float32
print(prediction.processed_images.shape)  # (N, H, W, 3) uint8
```

`model_name=` also works when you just want a preset without downloading
weights (e.g. for testing):

```python
model = DepthAnything3(model_name="da3-large")
```

### Pose-conditioned depth estimation

Providing extrinsics/intrinsics enables pose-conditioned mode, generally
improving depth consistency:

```python
prediction = model.inference(
    image=["image1.jpg", "image2.jpg"],
    extrinsics=extrinsics_array,  # (N, 4, 4)
    intrinsics=intrinsics_array,  # (N, 3, 3)
)
```

### Exporting results

```python
prediction = model.inference(
    image=image_paths,
    export_dir="./output",
    export_format="mini_npz-glb",  # combine formats with "-"
)
```

See {ref}`export-formats` above for every export format's own parameters
(passed via `export_kwargs`).

### Feature extraction

```python
prediction = model.inference(
    image=image_paths,
    export_dir="./output",
    export_format="feat_vis",
    export_feat_layers=[0, 1, 2],
)
```

### Gaussian Splatting export

Requires a model with the Gaussian branch (`da3-giant` or
`da3nested-giant-large`) and `infer_gs=True`:

```python
model = DepthAnything3(model_name="da3-giant").to("cuda")

prediction = model.inference(
    image=image_paths,
    extrinsics=extrinsics_array,
    intrinsics=intrinsics_array,
    export_dir="./output",
    export_format="npz-glb-gs_ply-gs_video",
    infer_gs=True,
)
```

### Ray-based pose estimation

```python
prediction = model.inference(
    image=image_paths,
    export_format="glb",
    use_ray_pose=True,
)
```

See {doc}`../model_cards` for the accuracy/speed tradeoff.

### Reference view selection

For multi-view input (3+ views), DA3 automatically picks which view anchors
the world coordinate frame. See {doc}`../ref_view_strategy` for the full
comparison of strategies; in short:

```python
# Default: balanced selection across views
prediction = model.inference(image_paths, ref_view_strategy="saddle_balanced")

# Temporally ordered input (e.g. video frames)
prediction = model.inference(video_frames, ref_view_strategy="middle")
```

### Custom model architectures

The model architecture is defined in
[`DepthAnything3Net`](https://github.com/bogarrichard/Depth-Anything-3/blob/dev/src/depth_anything_3/model/da3.py)
and configured via a YAML file under
[`src/depth_anything_3/configs`](https://github.com/bogarrichard/Depth-Anything-3/tree/dev/src/depth_anything_3/configs).
Input/output processing is handled by
{class}`~depth_anything_3.api.DepthAnything3`. To try a new architecture,
write a config file:

```yaml
__object__:
  path: depth_anything_3.model.da3
  name: DepthAnything3Net
  args: as_params

net:
  __object__:
    path: depth_anything_3.model.dinov2.dinov2
    name: DinoV2
    args: as_params

  name: vitb
  out_layers: [5, 7, 9, 11]
  alt_start: 4
  qknorm_start: 4
  rope_start: 4
  cat_token: True

head:
  __object__:
    path: depth_anything_3.model.dualdpt
    name: DualDPT
    args: as_params

  dim_in: &head_dim_in 1536
  output_dim: 2
  features: &head_features 128
  out_channels: &head_out_channels [96, 192, 384, 768]
```

Then build it:

```python
from depth_anything_3.cfg import create_object, load_config

model = create_object(load_config("path/to/new/config"))
```
