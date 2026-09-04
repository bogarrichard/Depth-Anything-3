# Depth Anything 3

```{note}
This is the documentation for [bogarrichard's fork](https://github.com/bogarrichard/Depth-Anything-3)
of Depth Anything 3 -- an unofficial, community-maintained build with
modernization work (dependency, tooling and packaging changes) on top of
the original project. It is not published or endorsed by ByteDance. For the
official upstream project, see
[ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3).
```

**Depth Anything 3 (DA3)** predicts spatially consistent geometry from
arbitrary visual inputs, with or without known camera poses. In pursuit of
minimal modeling, DA3 rests on two key insights:

- A **single plain transformer** (a vanilla DINO encoder) is sufficient as a
  backbone, without architectural specialization.
- A singular **depth-ray representation** removes the need for complex
  multi-task learning.

DA3 outperforms [Depth Anything 2](https://github.com/DepthAnything/Depth-Anything-V2)
for monocular depth estimation and [VGGT](https://github.com/facebookresearch/vggt)
for multi-view depth and pose estimation. All models are trained exclusively
on public academic datasets.

```{seealso}
[Paper (arXiv)](https://arxiv.org/abs/2511.10647) ·
[Project page](https://depth-anything-3.github.io) ·
[Hugging Face demo](https://huggingface.co/spaces/depth-anything/Depth-Anything-3) ·
[GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)
```

## Model Cards

| Model Name | Params | Rel. Depth | Pose Est. | Pose Cond. | GS | Met. Depth | Sky Seg | License |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Nested** | | | | | | | | |
| [DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1) | 1.40B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CC BY-NC 4.0 |
| **Any-view Model** | | | | | | | | |
| [DA3-GIANT-1.1](https://huggingface.co/depth-anything/DA3-GIANT-1.1) | 1.15B | ✅ | ✅ | ✅ | ✅ | | | CC BY-NC 4.0 |
| [DA3-LARGE-1.1](https://huggingface.co/depth-anything/DA3-LARGE-1.1) | 0.35B | ✅ | ✅ | ✅ | | | | CC BY-NC 4.0 |
| [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) | 0.12B | ✅ | ✅ | ✅ | | | | Apache 2.0 |
| [DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL) | 0.08B | ✅ | ✅ | ✅ | | | | Apache 2.0 |
| **Monocular Metric Depth** | | | | | | | | |
| [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE) | 0.35B | ✅ | | | | ✅ | ✅ | Apache 2.0 |
| **Monocular Depth** | | | | | | | | |
| [DA3MONO-LARGE](https://huggingface.co/depth-anything/DA3MONO-LARGE) | 0.35B | ✅ | | | | | ✅ | Apache 2.0 |

`DA3-LARGE-1.1` is generally comparable to VGGT. See {doc}`model_cards` for
the full table (including the deprecated pre-`1.1` checkpoints), a
model-choosing guide, the metric-depth conversion formula, and the ray head
vs. camera decoder tradeoff.

## Codebase highlights

- **Interactive web UI & gallery** -- visualize outputs with a Gradio-based
  interface (`da3 gradio`).
- **Flexible CLI** -- scriptable batch processing for images, image
  directories, video, and COLMAP datasets.
- **Multiple export formats** -- `glb`, `npz`, `ply`, 3DGS video, feature and
  depth visualizations, to connect with downstream tools.
- **Extensible, modular design** -- config-driven model construction (see
  {doc}`usage/python_api`) for research on new architectures.

## Where to go next

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
model_cards
```

```{toctree}
:maxdepth: 2
:caption: Usage

usage/python_api
usage/cli
```

```{toctree}
:maxdepth: 2
:caption: API reference

reference/python_api
reference/cli
```

```{toctree}
:maxdepth: 2
:caption: Guides

ref_view_strategy
benchmark
streaming
```

```{toctree}
:maxdepth: 1
:caption: Project

faq
```
