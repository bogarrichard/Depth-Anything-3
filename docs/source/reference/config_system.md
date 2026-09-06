# Config System

`depth_anything_3.cfg` builds model objects (and anything else) from plain
YAML: each object is a `__object__` mapping naming an import path and
constructor arguments, resolved recursively. See
{ref}`custom-model-architectures` in {doc}`../python_examples` for a
worked example.

```{eval-rst}
.. autofunction:: depth_anything_3.cfg.load_config

.. autofunction:: depth_anything_3.cfg.create_object
```
