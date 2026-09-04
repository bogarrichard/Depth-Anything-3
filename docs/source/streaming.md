# DA3-Streaming

Memory-efficient inference for long videos via chunk streaming.

`DA3-Streaming` lets Depth Anything 3 process **long video sequences** and
**super-large scenes** under tight CPU/GPU memory budgets, by chunking
frames and managing state across chunks. Built on the ideas of
[VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long), it focuses on memory
efficiency and stable, near-real-time online video inference.

```{note}
Released 2025-11-11. Lives under `da3_streaming/` in the repository, and
needs the `streaming` extra plus a git submodule -- see below.
```

## Setup

**1. Clone with the submodule.** `da3_streaming/loop_utils/salad` is a git
submodule this pipeline needs:

```bash
git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

Forgot `--recursive`?

```bash
cd <your_dir>/Depth-Anything-3/
git submodule update --init --recursive .
```

**2. Install dependencies:**

```bash
pip install -e ".[streaming]"
```

**3. Download pretrained weights:**

```bash
bash ./scripts/download_weights.sh
```

If you hit a `libGL.so.1` error (from `opencv-python`):

```bash
sudo apt-get install -y libgl1-mesa-glx
```

## Running

```bash
python da3_streaming.py --image_dir ./path_of_images
```

or with an explicit config and output directory:

```bash
python da3_streaming.py --image_dir ./path_of_images --config ./configs/base_config.yaml --output_dir ${OUTPUT_DIR}
```

Starting from a video instead of frames:

```bash
mkdir ./extract_images
ffmpeg -i your_video.mp4 -vf "fps=5,scale=640:-1" ./extract_images/frame_%06d.png
```

## Outputs

**Basic outputs:**

- `${OUTPUT_DIR}/camera_poses.txt` -- extrinsic matrix parameters, one frame
  per line.
- `${OUTPUT_DIR}/intrinsic.txt` -- `fx, fy, cx, cy` per frame.
- `${OUTPUT_DIR}/pcd/combined_pcd.ply` -- combined point cloud from all
  frames.

**Additional outputs** (with `save_depth_conf_result: True` in the config):

- `${OUTPUT_DIR}/results_output/` -- per-frame RGB, depth, confidence and
  intrinsics. Confidence's minimum value is 0.

Fuse a point cloud from `results_output/` to verify:

```bash
python npz_output_process.py \
    --npz_folder ${OUTPUT_DIR}/results_output \
    --pose_file ${OUTPUT_DIR}/camera_poses.txt \
    --output_file ${OUTPUT_DIR}/output.ply
```

```{warning}
Ensure sufficient disk space before running -- intermediate results are
deleted on completion to limit usage, but can be sizeable while running.
```

## Experiment results

`ATE RMSE [m]` on KITTI Odometry, comparing `DA3-Streaming`, `VGGT-Long` and
`Pi-Long`. All methods evaluated with overlap = half chunk size, comparable
resolution (~500px width), and loop closure at similarity threshold 0.85.

| Method | chunk size | AVG | AVG (w/o 01) | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VGGT-Long | 120 | 25.60 | 22.81 | 16.13 | 53.43 | 51.98 | 4.37 | 2.15 | 12.69 | 11.33 | 3.60 | 70.29 | 34.55 | 21.05 |
| Pi-Long | 120 | 21.17 | 11.81 | 5.55 | 114.83 | 50.29 | 1.63 | 1.11 | 3.48 | 2.88 | 3.92 | 24.25 | 7.38 | 17.61 |
| **DA3-Streaming** | 120 | 18.63 | **10.42** | 4.48 | 100.77 | 33.41 | 3.58 | 2.39 | 3.95 | 7.59 | 2.09 | 31.20 | 8.06 | 7.44 |
| VGGT-Long | 60 | 26.36 | 19.30 | 8.06 | 96.96 | 34.16 | 6.83 | 4.16 | 9.15 | 4.68 | 2.68 | 63.15 | 32.24 | 27.87 |
| Pi-Long | 60 | 30.63 | 17.10 | 7.82 | 165.92 | 73.59 | 3.67 | 0.91 | 5.16 | 3.89 | 3.57 | 33.97 | 17.01 | 21.41 |
| **DA3-Streaming** | 60 | **16.83** | 10.64 | 5.13 | 78.76 | 35.64 | 5.38 | 3.18 | 3.04 | 2.83 | 2.32 | 26.55 | 8.86 | 13.42 |

The pipeline is restructured and GPU-accelerated, reaching nearly **10 FPS**
without the keyframe strategy (excluding warm-up, model loading, and PLY
saving). Measured on KITTI sequences 00, 05, 08 (11,373 frames total) on an
NVIDIA A100:

| Method | Time | FPS |
|---|---|---|
| VGGT-Long | 65min 08sec | 2.91 |
| Pi-long | 60min 09sec | 3.15 |
| **DA3-Streaming** | 22min 17sec | **8.51** |

Not an SLAM system, but still competitive with uncalibrated SLAM on TUM
RGB-D (chunk size 120, overlap 60, ~500px width, with loop closure):

| Method | AVG | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz |
|---|---|---|---|---|---|---|---|---|---|---|
| Droid-SLAM Uncalibrated | 0.163 | 0.202 | 0.032 | 0.091 | 0.064 | 0.045 | 0.918 | 0.056 | 0.045 | 0.012 |
| Mast3r-SLAM Uncalibrated | **0.060** | 0.070 | 0.035 | 0.055 | 0.056 | 0.035 | 0.118 | 0.041 | 0.114 | 0.020 |
| VGGT-Long | 0.110 | 0.118 | 0.058 | 0.111 | 0.118 | 0.071 | 0.155 | 0.140 | 0.120 | 0.099 |
| Pi-long | 0.094 | 0.115 | 0.047 | 0.052 | 0.160 | 0.085 | 0.114 | 0.143 | 0.081 | 0.052 |
| **DA3-Streaming** | 0.087 | 0.059 | 0.034 | 0.042 | 0.107 | 0.060 | 0.105 | 0.206 | 0.126 | 0.044 |

Chunk-size sweep (KITTI w/o 01 at 504×154, TUM RGB-D at 504×378, overlap =
half chunk size):

| | Chunk size | 120 | 90 | 60 | 30 |
|---|---|---|---|---|---|
| KITTI (504×154) | Peak VRAM [GB] | 15.9 | 14.3 | 12.7 | 11.5 |
| | ATE RMSE [m] | 10.42 | 9.38 | 10.64 | 19.39 |
| TUM RGB-D (504×378) | Peak VRAM [GB] | 28.3 | 25.1 | 21.2 | 18.7 |
| | ATE RMSE [m] | 0.087 | 0.091 | 0.127 | 0.227 |

## Acknowledgements

Built on [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) and
Depth Anything 3.
