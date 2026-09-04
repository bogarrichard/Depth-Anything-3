# Reference View Selection Strategy

## Overview

Reference view selection is a component of multi-view depth estimation.
When processing multiple input views, the model must decide which view
serves as the primary reference frame defining the world coordinate system.

A different reference view leads to different reconstruction results -- a
known consideration in multi-view geometry, analyzed in
[PI3](https://arxiv.org/abs/2507.13347). The choice can affect the quality
and consistency of depth predictions across the scene.

## Automatic reference view selection

DA3 addresses this with **automatic reference view selection** based on
**class tokens**: instead of a fixed heuristic or manual choice, the model
analyzes the class-token features from all input views and picks the most
suitable reference frame.

## Available strategies

### `saddle_balanced` (recommended, default)

Selects a view that balances multiple feature metrics -- a "middle ground"
view that is neither too similar nor too different from the others, making
it a stable reference point.

1. Extract and normalize class tokens from all views.
2. Compute three complementary metrics per view: average cosine similarity
   to other views, feature norm, and feature variance.
3. Normalize each metric to `[0, 1]`.
4. Select the view closest to 0.5 (median) across all three.

### `saddle_sim_range`

Selects the view with the largest similarity range to other views --
"saddle point" views that are highly similar to some views and dissimilar
to others, making them information-rich anchors.

1. Compute pairwise cosine similarity between all views.
2. For each view, compute the range (max − min) of similarities to others.
3. Select the view with the maximum range.

### `first` (not recommended)

Always uses the first view (index 0). Only use this if you've manually
pre-sorted your views and know the first one is optimal, or for debugging /
baseline comparisons.

### `middle`

Uses the view at index `S // 2` (`S` = number of views). Recommended
**only when input images are temporally ordered** -- video sequences (e.g.
the DA3-LONG setting) or other sequential captures, where the middle frame
has maximum overlap with all other frames and the most stable viewpoint.

## Usage

### Python API

```python
from depth_anything_3 import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")

# Default
prediction = model.inference(images, ref_view_strategy="saddle_balanced")

# Temporally ordered input
prediction = model.inference(video_frames, ref_view_strategy="middle")

# Wide-baseline multi-view
prediction = model.inference(images, ref_view_strategy="saddle_sim_range")
```

### CLI

```bash
da3 auto input/ --export-dir output/                         # default
da3 auto input/ --ref-view-strategy saddle_balanced           # explicit
da3 video input.mp4 --ref-view-strategy middle
da3 images captures/ --ref-view-strategy saddle_sim_range
```

### When selection is applied

Only when the number of views `S >= 3`. For 1-2 views, no reordering
happens (equivalent to `first`).

## Recommendations

| Scenario | Strategy | Why |
|---|---|---|
| Default / unknown | `saddle_balanced` | Robust across diverse scenarios |
| Video frames | `middle` | Temporal coherence, stable middle frame |
| Wide-baseline multi-view | `saddle_sim_range` | Maximizes information coverage |
| Pre-sorted inputs | `first` | Only if ordering was manually optimized |
| Single image | `first` | Automatic, no reordering needed for `S <= 2` |

Start with the default, switch to `middle` for video, and experiment if
results look suboptimal -- check `glb` quality and consistency across
views.

## Technical details

Selection triggers only when `num_views >= 3`. It happens at layer
`alt_start - 1` in the vision transformer, before the first global
attention layer, so the selected reference view influences the entire depth
prediction pipeline. The overhead is negligible.

## FAQ

**Why is this feature provided?**
The model can handle any view order, but automatic reference view selection
can improve depth prediction quality in multi-view scenarios.

**Does this add computational cost?**
The overhead is negligible.

**Can I manually specify which view to use as reference?**
Not directly through this parameter -- pre-sort your input images to place
the preferred view first and use `ref_view_strategy="first"`.

**What happens if I don't specify this parameter?**
The default, `saddle_balanced`, is used.

**Is this feature used in the DA3 paper benchmarks?**
No -- the paper used `first` as the default for all multi-view experiments.
The current default was updated to `saddle_balanced` for better robustness.
