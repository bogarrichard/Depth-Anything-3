# Python API reference

Generated from the source docstrings -- always in sync with the installed
version. See {doc}`../usage/python_api` for a task-oriented walkthrough with
runnable examples.

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
[`utils/export/colmap.py`](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/utils/export/colmap.py)
on GitHub.
```

## Config system

```{eval-rst}
.. autofunction:: depth_anything_3.cfg.load_config

.. autofunction:: depth_anything_3.cfg.create_object
```
