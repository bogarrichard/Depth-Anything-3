# Quickstart

Assumes the package is already installed -- see {doc}`installation` if not.

## Python

```python
from depth_anything_3.api import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cuda")
prediction = model.inference(["image1.png", "image2.png"])

print(prediction.depth.shape)       # (N, H, W)  float32
print(prediction.extrinsics.shape)  # (N, 3, 4)  float32, opencv world-to-camera
```

`prediction` is a {class}`~depth_anything_3.specs.Prediction` dataclass;
{doc}`reference/python_api` documents every field.

## Command line

```bash
da3 auto path/to/input --export-dir ./output
```

`da3 auto` detects whether the input is an image, an image directory, a
video, or a COLMAP dataset, and dispatches accordingly. The export
directory then holds `scene.glb`, `scene.jpg`, and whatever else the
requested `--export-format` produces.

## Without installing anything

- Run [`notebooks/da3.ipynb`](https://colab.research.google.com/github/bogarrichard/Depth-Anything-3/blob/dev/notebooks/da3.ipynb)
  on a free Colab GPU.
- Open the repository in
  [GitHub Codespaces](https://codespaces.new/bogarrichard/Depth-Anything-3)
  for a full (CPU-only) dev environment, then run `uv run python demo.py`.

## Next steps

- {doc}`model_cards` -- pick a checkpoint for your accuracy/speed budget.
- {doc}`python_api_examples` -- pose-conditioned inference, exports,
  Gaussian Splatting, feature extraction.
- {doc}`cli_examples` -- batch processing, the resident backend, the
  Gradio UI and gallery.
