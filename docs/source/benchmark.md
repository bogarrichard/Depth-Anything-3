# Benchmark Evaluation

Instructions for running the Visual Geometry Benchmark evaluation pipeline
shipped with Depth Anything 3.

## Highlights

- **Diverse, challenging datasets** -- 5 datasets (ETH3D, 7Scenes,
  ScanNet++, HiRoom, DTU) spanning objects to indoor/outdoor scenes. Some
  are recalibrated for higher accuracy (see {ref}`scannetpp-details`
  below). All preprocessed datasets are on
  [depth-anything/DA3-BENCH](https://huggingface.co/datasets/depth-anything/DA3-BENCH).
- **Robust evaluation pipeline** -- RANSAC-based pose alignment, TSDF fusion
  for reconstruction.
- **Standardized metrics** -- AUC for pose accuracy, F1-score and Chamfer
  Distance for reconstruction.

Install the `bench` extra first -- see {doc}`installation`.

## Quick start

Install the HuggingFace CLI first: `pip install -U huggingface_hub[cli]`. If
downloads are slow, try `export HF_ENDPOINT=https://hf-mirror.com`.

```bash
cd da3_release

mkdir -p workspace/benchmark_dataset
hf download depth-anything/DA3-BENCH \
    --local-dir workspace/benchmark_dataset \
    --repo-type dataset

cd workspace/benchmark_dataset
for f in *.zip; do unzip -q "$f"; done
```

```bash
MODEL=depth-anything/DA3-GIANT

# Full evaluation (all datasets, all modes)
python -m depth_anything_3.bench.evaluator model.path=$MODEL

# View results
python -m depth_anything_3.bench.evaluator eval.print_only=true
```

## Dataset download

All datasets are hosted at
[depth-anything/DA3-BENCH](https://huggingface.co/datasets/depth-anything/DA3-BENCH):

| Dataset | File | Size | Description |
|---|---|---|---|
| ETH3D | `eth3d.zip` | ~14.1 GB | High-resolution multi-view stereo (indoor/outdoor) |
| ScanNet++ | `scannetpp.zip` | ~10.1 GB | High-quality RGB-D indoor scenes |
| DTU-49 | `dtu.zip` | ~8.3 GB | Multi-view stereo benchmark (22 scenes × 49 views) |
| 7Scenes | `7scenes.zip` | ~3.3 GB | RGB-D indoor localization |
| DTU-64 | `dtu64.zip` | ~1.7 GB | DTU subset for pose evaluation (13 scenes × 64 views) |
| HiRoom | `hiroom.zip` | ~0.7 GB | High-resolution indoor rooms |

Download all of it, one file, or grab the zips manually from the dataset
page:

```bash
# All
hf download depth-anything/DA3-BENCH --local-dir workspace/benchmark_dataset --repo-type dataset

# Just HiRoom
hf download depth-anything/DA3-BENCH hiroom.zip --local-dir workspace/benchmark_dataset --repo-type dataset
```

```bash
cd workspace/benchmark_dataset
for f in *.zip; do unzip -q "$f"; done   # all
unzip hiroom.zip                          # or just one
```

Expected layout after extraction:

```text
workspace/benchmark_dataset/
├── eth3d/
│   ├── courtyard/
│   └── ...
├── 7scenes/
│   └── 7Scenes/
│       ├── chess/
│       └── ...
├── scannetpp/
│   ├── 09c1414f1b/
│   └── ...
├── hiroom/
│   ├── data/
│   ├── fused_pcd/
│   └── selected_scene_list_val.txt
├── dtu/
│   ├── Rectified/
│   ├── Cameras/
│   ├── Points/
│   ├── SampleSet/
│   └── depth_raw/
└── dtu64/
    ├── Cameras/
    ├── scan105/
    └── ...
```

## Evaluation pipeline

| Mode | Description | Metrics |
|---|---|---|
| `pose` | Camera pose estimation | AUC@3°, AUC@30° |
| `recon_unposed` | 3D reconstruction with **predicted** poses | F-score, Overall |
| `recon_posed` | 3D reconstruction with **GT** poses | F-score, Overall |

```bash
cd da3_release
MODEL=depth-anything/DA3-GIANT

python -m depth_anything_3.bench.evaluator model.path=$MODEL              # full run
python -m depth_anything_3.bench.evaluator eval.eval_only=true            # skip inference
python -m depth_anything_3.bench.evaluator eval.print_only=true           # just print saved metrics
```

Selective evaluation:

```bash
python -m depth_anything_3.bench.evaluator model.path=$MODEL eval.datasets=[hiroom]
python -m depth_anything_3.bench.evaluator model.path=$MODEL eval.modes=[pose,recon_unposed]
python -m depth_anything_3.bench.evaluator model.path=$MODEL eval.datasets=[hiroom] eval.modes=[pose]
```

Multi-GPU inference is automatic; scope it with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m depth_anything_3.bench.evaluator model.path=$MODEL
```

## Configuration

Default config: `src/depth_anything_3/bench/configs/eval_bench.yaml`.

```yaml
model:
  path: depth-anything/DA3-GIANT

workspace:
  work_dir: ./workspace/evaluation

eval:
  datasets: [eth3d, 7scenes, scannetpp, hiroom, dtu, dtu64]
  modes: [pose, recon_unposed, recon_posed]
  max_frames: 100      # -1 = no limit
  scenes: null         # null = all

inference:
  num_fusion_workers: 4
  debug: false
```

Output layout:

```text
workspace/evaluation/
├── model_results/              # inference outputs
│   ├── eth3d/{scene}/{unposed,posed}/
│   └── ...
└── metric_results/             # metrics, as JSON
    ├── eth3d_pose.json
    ├── eth3d_recon_unposed.json
    └── ...
```

## Metrics

| Metric | Description |
|---|---|
| Auc3 | Area under curve at 3° angular error threshold |
| Auc30 | Area under curve at 30° angular error threshold |
| F-score | Harmonic mean of precision and recall (higher is better) |
| Overall | (Accuracy + Completeness) / 2, error in meters/mm (lower is better) |

DTU reports Overall in millimeters; other datasets report in meters.

### Expected results for DA3-GIANT

```text
========================================================
SUMMARY
========================================================

POSE ESTIMATION
---------------------------------------------------------------------------------------
Metric         Avg         HiRoom      ETH3D       DTU-64      7Scenes     ScanNet++
---------------------------------------------------------------------------------------
Auc3           0.6705      0.8030      0.4872      0.9408      0.2744      0.8470
Auc30          0.9436      0.9592      0.9153      0.9939      0.8668      0.9827

RECON_UNPOSED (Pred Pose)
---------------------------------------------------------------------------------------
Metric         Avg*        HiRoom      ETH3D       DTU         7Scenes     ScanNet++
---------------------------------------------------------------------------------------
F-score        0.7345      0.8629      0.7876      N/A         0.5043      0.7831
Overall        0.1682      0.0457      0.4366      1.7927      0.1230      0.0676

RECON_POSED (GT Pose)
---------------------------------------------------------------------------------------
Metric         Avg*        HiRoom      ETH3D       DTU         7Scenes     ScanNet++
---------------------------------------------------------------------------------------
F-score        0.7978      0.9546      0.8685      N/A         0.5635      0.8045
Overall        0.1408      0.0213      0.3679      1.7488      0.1092      0.0649

* Avg F-score / Overall = average over HiRoom, ETH3D, 7Scenes, ScanNet++ (4 datasets)
```

## Dataset details

### ETH3D

High-resolution multi-view stereo with laser-scanned ground truth. 11
scenes, variable resolution. Some images with unusual camera rotations are
filtered out (see `ETH3D_FILTER_KEYS` in `constants.py`).

### 7Scenes

RGB-D camera relocalization dataset. 7 scenes, 640×480, poses from
KinectFusion, meshes from TSDF fusion.

(scannetpp-details)=
### ScanNet++

High-quality indoor RGB-D dataset, 20 validation scenes, 768×1024 after
undistortion, ground truth from a FARO scanner. The default poses are often
inaccurate (motion blur, textureless iPhone frames), so these were re-run
through COLMAP with blurry-frame filtering, joint fisheye calibration, and
exhaustive matching (several days per scene, but necessary for quality).
Processed scenes:
[haotongl/scannetpp_zipnerf](https://huggingface.co/datasets/haotongl/scannetpp_zipnerf).

### HiRoom

Indoor room scenes, 24 validation scenes, ground truth from fused point
clouds.

### DTU-49 (reconstruction only)

MVSNet evaluation protocol, 22 scenes, 49 views/scene, laser-scanned point
clouds with observation masks, Overall-only metric.

### DTU-64 (pose only)

13 scenes, 64 views/scene, AUC@3°/AUC@30°. Two DTU settings exist because
more views make pose estimation harder (DTU-64), while DTU-49 follows the
standard MVS comparison protocol.

## Command reference

```text
python -m depth_anything_3.bench.evaluator [OPTIONS] [KEY=VALUE ...]

Configuration:
  --config PATH                      Config YAML file (default: bench/configs/eval_bench.yaml)

Config overrides (dotlist notation):
  model.path=VALUE                   Model path or HuggingFace ID
  workspace.work_dir=VALUE           Working directory for outputs
  eval.datasets=[dataset1,dataset2]  eth3d,7scenes,scannetpp,hiroom,dtu,dtu64
  eval.modes=[mode1,mode2]           pose,recon_unposed,recon_posed
  eval.scenes=[scene1,scene2]        Specific scenes (null=all)
  eval.max_frames=VALUE              Max frames per scene (-1=no limit, default: 100)
  eval.ref_view_strategy=VALUE       Reference view strategy (default: first)
  eval.eval_only=VALUE               Only run evaluation (skip inference)
  eval.print_only=VALUE              Only print saved metrics
  inference.num_fusion_workers=VALUE Number of parallel workers (default: 4)
  inference.debug=VALUE              Enable debug mode

Multi-GPU: use CUDA_VISIBLE_DEVICES (auto-detected and distributed otherwise).
```

```bash
MODEL=depth-anything/DA3-GIANT

# Quick test on HiRoom only
python -m depth_anything_3.bench.evaluator model.path=$MODEL eval.datasets=[hiroom] eval.modes=[pose]

# Pose-only, all 5 pose datasets
python -m depth_anything_3.bench.evaluator model.path=$MODEL \
    eval.datasets=[eth3d,7scenes,scannetpp,hiroom,dtu64] eval.modes=[pose]

# Recon-only, all 5 recon datasets
python -m depth_anything_3.bench.evaluator model.path=$MODEL \
    eval.datasets=[eth3d,7scenes,scannetpp,hiroom,dtu] eval.modes=[recon_unposed,recon_posed]

# Debug a specific scene
python -m depth_anything_3.bench.evaluator model.path=$MODEL \
    eval.datasets=[eth3d] eval.scenes=[courtyard] inference.debug=true
```

## Troubleshooting

Check dataset paths in `src/depth_anything_3/utils/constants.py`:

```python
ETH3D_EVAL_DATA_ROOT = "workspace/benchmark_dataset/eth3d"
SEVENSCENES_EVAL_DATA_ROOT = "workspace/benchmark_dataset/7scenes"
SCANNETPP_EVAL_DATA_ROOT = "workspace/benchmark_dataset/scannetpp"
HIROOM_EVAL_DATA_ROOT = "workspace/benchmark_dataset/hiroom/data"
DTU_EVAL_DATA_ROOT = "workspace/benchmark_dataset/dtu"
DTU64_EVAL_DATA_ROOT = "workspace/benchmark_dataset/dtu64"
```

## Citation

```bibtex
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}
```

Please also cite the original dataset papers for each benchmark you use.

## License

Benchmark datasets are for research purposes only -- follow each dataset's
own license:

- [ETH3D](https://www.eth3d.net/)
- [7Scenes (Microsoft Research)](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [ScanNet++](http://www.scan-net.org/)
- [DTU](https://roboimagedata.compute.dtu.dk/)
- [HiRoom (SVLightVerse)](https://jerrypiglet.github.io/SVLightVerse/)
