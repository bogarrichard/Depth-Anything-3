<div align="center">
<h1 style="border-bottom: none; margin-bottom: 0px ">Depth Anything 3: Recovering the Visual Space from Any Views</h1>

[**Haotong Lin**](https://haotongl.github.io/)<sup>&ast;</sup> · [**Sili Chen**](https://github.com/SiliChen321)<sup>&ast;</sup> · [**Jun Hao Liew**](https://liewjunhao.github.io/)<sup>&ast;</sup> · [**Donny Y. Chen**](https://donydchen.github.io)<sup>&ast;</sup> · [**Zhenyu Li**](https://zhyever.github.io/) · [**Guang Shi**](https://scholar.google.com/citations?user=MjXxWbUAAAAJ&hl=en) · [**Jiashi Feng**](https://scholar.google.com.sg/citations?user=Q8iay0gAAAAJ&hl=en)
<br>
[**Bingyi Kang**](https://bingyikang.com/)<sup>&ast;&dagger;</sup>

&dagger;project lead&emsp;&ast;Equal Contribution

<a href="https://arxiv.org/abs/2511.10647"><img src='https://img.shields.io/badge/arXiv-Depth Anything 3-red' alt='Paper PDF'></a>
<a href='https://depth-anything-3.github.io'><img src='https://img.shields.io/badge/Project_Page-Depth Anything 3-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/depth-anything/Depth-Anything-3'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<a href='https://bogarrichard.github.io/Depth-Anything-3/'><img src='https://img.shields.io/badge/docs-GitHub%20Pages-blue'></a>
<a href='https://github.com/bogarrichard/Depth-Anything-3/actions/workflows/ci.yml'><img src='https://github.com/bogarrichard/Depth-Anything-3/actions/workflows/ci.yml/badge.svg?branch=dev' alt='CI'></a>
<a href='https://colab.research.google.com/github/bogarrichard/Depth-Anything-3/blob/dev/notebooks/da3.ipynb'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Open In Colab'></a>
<a href='https://codespaces.new/bogarrichard/Depth-Anything-3?ref=dev'><img src='https://github.com/codespaces/badge.svg' alt='Open in GitHub Codespaces'></a>

</div>

**Depth Anything 3 (DA3)** predicts spatially consistent geometry from
arbitrary visual inputs, with or without known camera poses. A single plain
transformer backbone and a singular depth-ray representation outperform
[DA2](https://github.com/DepthAnything/Depth-Anything-V2) for monocular
depth and [VGGT](https://github.com/facebookresearch/vggt) for multi-view
depth and pose estimation, trained exclusively on public academic datasets.

<p align="center">
  <img src="assets/images/demo320-2.gif" alt="Depth Anything 3 - Left" width="70%">
</p>
<p align="center">
  <img src="assets/images/da3_radar.png" alt="Depth Anything 3" width="100%">
</p>

## 📚 Full documentation: [bogarrichard.github.io/Depth-Anything-3](https://bogarrichard.github.io/Depth-Anything-3/)

Installation (uv/pip), model cards, Python API and CLI usage and reference,
the benchmark pipeline, DA3-Streaming, and FAQ all live there. Published
automatically from `docs/source` on every push to `dev`
(`.github/workflows/docs.yml`).

## 📰 News

- **11-12-2025:** New models and [DA3-Streaming](https://bogarrichard.github.io/Depth-Anything-3/streaming.html) released -- ultra-long video inference under 12GB GPU memory via sliding-window streaming.
- **08-12-2025:** [Benchmark evaluation pipeline](https://bogarrichard.github.io/Depth-Anything-3/benchmark.html) released.
- **30-11-2025:** Added `use_ray_pose` and [`ref_view_strategy`](https://bogarrichard.github.io/Depth-Anything-3/ref_view_strategy.html).
- **25-11-2025:** Added [Awesome DA3 Projects](#-awesome-da3-projects).
- **14-11-2025:** Paper, project page, code and models released.

## 🚀 Quick start

```bash
uv sync            # or: pip install -e .
```

```python
from depth_anything_3.api import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cuda")
prediction = model.inference(["image1.png", "image2.png"])
```

```bash
da3 auto path/to/input --export-dir ./output
```

**Try it without installing anything locally:**
- Click **Open In Colab** above to run [`notebooks/da3.ipynb`](notebooks/da3.ipynb) on a free GPU.
- Click **Open in GitHub Codespaces** above for a full dev environment
  (CPU only), then run `uv run python demo.py`.

See the [installation guide](https://bogarrichard.github.io/Depth-Anything-3/installation.html)
for extras (Gradio app, Gaussian Splatting, COLMAP export, streaming), and
the [model cards](https://bogarrichard.github.io/Depth-Anything-3/model_cards.html)
to pick a checkpoint.

## 🏢 Awesome DA3 Projects

A community-curated list of Depth Anything 3 integrations. Submit yours via
PR.

- [DA3-blender](https://github.com/xy-gao/DA3-blender) -- Blender addon for DA3-based 3D reconstruction.
- [ComfyUI-DepthAnythingV3](https://github.com/PozzettiAndrea/ComfyUI-DepthAnythingV3) -- ComfyUI nodes for DA3.
- [DA3-ROS2-Wrapper](https://github.com/GerdsenAI/GerdsenAI-Depth-Anything-3-ROS2-Wrapper) -- Real-time DA3 depth in ROS2.
- [DA3-ROS2-CPP-TensorRT](https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt) -- DA3 ROS2 C++ TensorRT inference node.
- [VideoDepthViewer3D](https://github.com/amariichi/VideoDepthViewer3D) -- DA3 metric depth streamed to a Three.js/WebXR viewer.

## 🧑‍💻 Official Codebase Core Contributors and Maintainers

<table>
  <tr>
    <td align="center">
      <a href="https://bingykang.github.io/">
        <img src="https://images.weserv.nl/?url=https://bingykang.github.io/images/bykang_homepage.jpeg?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub><b>Bingyi Kang</b></sub>
    </td>
    <td align="center">
      <a href="https://haotongl.github.io/">
        <img src="https://images.weserv.nl/?url=https://haotongl.github.io/assets/img/prof_pic.jpg?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub>Haotong Lin</sub>
    </td>
    <td align="center">
      <a href="https://github.com/SiliChen321">
        <img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/195901058?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub>Sili Chen</sub>
    </td>
    <td align="center">
      <a href="https://liewjunhao.github.io/">
        <img src="https://images.weserv.nl/?url=https://liewjunhao.github.io/images/liewjunhao.png?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
       </a>
        <br />
        <sub>Jun Hao Liew</sub>
    </td>
    <td align="center">
      <a href="https://donydchen.github.io/">
        <img src="https://images.weserv.nl/?url=https://donydchen.github.io/assets/img/profile.jpg?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub>Donny Y. Chen</sub>
    </td>
    <td align="center">
      <a href="https://github.com/DengKaiCQ">
        <img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/59907452?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub>Kai Deng</sub>
    </td>
  </tr>
</table>

## 📝 Citation

```bibtex
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}
```
