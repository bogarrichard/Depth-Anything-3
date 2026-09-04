# Depth Anything 3

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
