# FAQ

**How do I get metric depth in meters?**
See the "Metric depth conversion" section of {doc}`model_cards`.

**What's `use_ray_pose` and when should I set it?**
See the "Ray head vs. camera decoder" section of {doc}`model_cards`.

**Which reference view does the model use for multi-view input, and can I
control it?**
See {doc}`ref_view_strategy`.

**Where's the full list of CLI flags / Python parameters?**
{doc}`reference/cli` and {doc}`reference/python_api` -- both generated
directly from the code, so they can't go stale.

**I found a bug, or want to contribute.**
Open an issue or pull request against
[this fork](https://github.com/bogarrichard/Depth-Anything-3) -- it's what
this documentation describes (packaging, tooling, and dependency changes on
top of the original project). For the model/architecture itself, the
[upstream project](https://github.com/ByteDance-Seed/Depth-Anything-3) is
the canonical place.
