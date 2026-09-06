# Installation

Requires Python 3.10–3.13.

```{note}
This page documents [bogarrichard's fork](https://github.com/bogarrichard/Depth-Anything-3)
specifically -- the `uv`/lockfile workflow, extras layout, and dependency
versions below are this fork's, and may not match the upstream
[ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
project. Clone this fork, not upstream, to follow these instructions:
```

```bash
git clone https://github.com/bogarrichard/Depth-Anything-3.git
cd Depth-Anything-3
```

## Basic install

`````{tab-set}

````{tab-item} uv
[uv](https://docs.astral.sh/uv/) installs from the tracked lockfile, so you
get the exact resolved versions this project is tested against:

```bash
uv sync
```

`[tool.uv] torch-backend = "auto"` picks the right torch wheel for the
machine it runs on, so no CUDA variant is baked into the lockfile.
````

````{tab-item} pip
```bash
pip install -e .
```
````

`````

## Installing with extras

Substitute any extra from the table below for `<extra>`; repeat the flag
(uv) or comma-separate (pip) to combine several.

`````{tab-set}

````{tab-item} uv
```bash
uv sync --extra <extra>          # one extra
uv sync --extra app --extra gs   # several
uv sync --extra all              # every user-facing extra
uv sync --all-extras             # ...plus dev and docs
```
````

````{tab-item} pip
```bash
pip install -e ".[<extra>]"      # one extra
pip install -e ".[app,gs]"       # several
pip install -e ".[all]"          # every user-facing extra
pip install -e ".[all,dev,docs]" # ...plus dev and docs
```
````

`````

| `<extra>` | Adds | Needed for |
|---|---|---|
| `app` | `gradio`, `pillow-heif` | `da3 gradio` web UI |
| `gs` | `gsplat`, `e3nn` | 3D Gaussian Splatting inference/export |
| `colmap` | `pycolmap` | `--export-format colmap` |
| `bench` | `open3d`, `scikit-learn` | `depth_anything_3.bench` evaluation |
| `streaming` | `faiss-gpu`, `numba`, `pandas`, ... | `da3_streaming/` (see {doc}`streaming`) |
| `dev` | `pytest`, `hypothesis`, `pre-commit` | running the test suite |
| `docs` | `sphinx`, `myst-parser`, ... | building this site |

`all` bundles every user-facing extra except `dev` and `docs`.

For DA3-Streaming specifically, the plain clone above isn't enough -- it
needs a git submodule too. Either clone with `--recursive` from the start,
or run `git submodule update --init --recursive` in an existing checkout.
See {doc}`streaming` for the full setup.

## Next step

See {doc}`model_cards` to pick a model, then {doc}`reference/python_api` or
{doc}`reference/cli` to run it.
