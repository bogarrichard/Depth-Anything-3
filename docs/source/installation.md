# Installation

Requires Python 3.10–3.13.

`````{tab-set}

````{tab-item} uv
[uv](https://docs.astral.sh/uv/) installs from the tracked lockfile, so you
get the exact resolved versions this project is tested against:

```bash
uv sync                    # Basic
uv sync --extra app        # + Gradio app
uv sync --extra gs         # + gaussian head (gsplat)
uv sync --extra streaming  # + da3_streaming pipeline
uv sync --extra colmap     # + COLMAP export (--export-format colmap)
uv sync --extra bench      # + benchmark evaluation pipeline
uv sync --all-extras       # ALL
```

`[tool.uv] torch-backend = "auto"` picks the right torch wheel for the
machine it runs on, so no CUDA variant is baked into the lockfile.
````

````{tab-item} pip
```bash
pip install -e .              # Basic
pip install -e ".[app]"       # Gradio app
pip install -e ".[gs]"        # gaussian head (gsplat)
pip install -e ".[streaming]" # da3_streaming pipeline
pip install -e ".[colmap]"    # COLMAP export (--export-format colmap)
pip install -e ".[bench]"     # benchmark evaluation pipeline
pip install -e ".[all]"       # ALL
```
````

`````

## Extras at a glance

| Extra | Adds | Needed for |
|---|---|---|
| `app` | `gradio`, `pillow-heif` | `da3 gradio` web UI |
| `gs` | `gsplat`, `e3nn` | 3D Gaussian Splatting inference/export |
| `colmap` | `pycolmap` | `--export-format colmap` |
| `bench` | `open3d`, `scikit-learn` | `depth_anything_3.bench` evaluation |
| `streaming` | `faiss-gpu`, `numba`, `pandas`, ... | `da3_streaming/` (see {doc}`streaming`) |
| `dev` | `pytest`, `hypothesis`, `pre-commit` | running the test suite |
| `docs` | `sphinx`, `myst-parser`, ... | building this site |

`all` bundles every user-facing extra except `dev` and `docs`.

For DA3-Streaming specifically, clone the repository with `--recursive` --
it pulls in a git submodule the streaming pipeline needs. See
{doc}`streaming` for the full setup.

## Next step

See {doc}`model_cards` to pick a model, then {doc}`usage/python_api` or
{doc}`usage/cli` to run it.
