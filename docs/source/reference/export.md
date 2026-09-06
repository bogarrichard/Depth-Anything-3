# Export Options

Every export format is one function below, selected by name in
`export_format` (combine several with `-`, e.g. `"mini_npz-glb"`) and
dispatched by `export()`. Format-specific parameters (`export_kwargs`) are
that function's own keyword arguments. `export_format` is used the same way
from both {doc}`high_level_api` and {doc}`cli` -- see
{doc}`../python_examples` for it in context.

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
