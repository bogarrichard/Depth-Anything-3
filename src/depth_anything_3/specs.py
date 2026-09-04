# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import torch


@dataclass
class Gaussians:
    """3D Gaussian Splatting parameters, all in world space.

    Attributes:
        means: World-space centers, shape ``(batch, gaussian, 3)``.
        scales: Standard-deviation scales, shape ``(batch, gaussian, 3)``.
        rotations: World-space orientation as scalar-first (WXYZ)
            quaternions, shape ``(batch, gaussian, 4)``.
        harmonics: Spherical-harmonic color coefficients, shape
            ``(batch, gaussian, 3, d_sh)``.
        opacities: Per-Gaussian opacity, shape ``(batch, gaussian)``, or its
            own SH coefficients, shape ``(batch, gaussian, 1, d_sh)``.
    """

    means: torch.Tensor  # world points, "batch gaussian dim"
    scales: torch.Tensor  # scales_std, "batch gaussian 3"
    rotations: torch.Tensor  # world_quat_wxyz, "batch gaussian 4"
    harmonics: torch.Tensor  # world SH, "batch gaussian 3 d_sh"
    opacities: torch.Tensor  # opacity | opacity SH, "batch gaussian" | "batch gaussian 1 d_sh"


@dataclass
class Prediction:
    """The result of :meth:`depth_anything_3.api.DepthAnything3.inference`.

    Attributes:
        depth: Estimated depth maps, shape ``(N, H, W)``.
        is_metric: Whether ``depth`` is in real-world units (metric models)
            rather than an arbitrary relative scale.
        sky: Sky segmentation mask, shape ``(N, H, W)``, if the model
            supports it.
        conf: Per-pixel confidence, shape ``(N, H, W)``.
        extrinsics: World-to-camera matrices, shape ``(N, 4, 4)``.
        intrinsics: Camera intrinsic matrices, shape ``(N, 3, 3)``.
        processed_images: The resized/normalized input images actually fed
            to the model, shape ``(N, H, W, 3)`` -- useful for visualization
            alongside ``depth``, since it matches its resolution.
        gaussians: Predicted 3D Gaussians, if ``infer_gs=True`` was passed to
            ``inference()``.
        aux: Auxiliary outputs, e.g. ``feat_layer_<i>`` intermediate
            features when ``export_feat_layers`` was set.
        scale_factor: The metric scale factor applied, for metric models.
    """

    depth: np.ndarray  # N, H, W
    is_metric: int
    sky: np.ndarray | None = None  # N, H, W
    conf: np.ndarray | None = None  # N, H, W
    extrinsics: np.ndarray | None = None  # N, 4, 4
    intrinsics: np.ndarray | None = None  # N, 3, 3
    processed_images: np.ndarray | None = None  # N, H, W, 3 - processed images for visualization
    gaussians: Gaussians | None = None  # 3D gaussians
    aux: dict[str, Any] = None  #
    scale_factor: Optional[float] = None  # metric scale
