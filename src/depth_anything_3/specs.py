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
    """A dataclass holding 3D Gaussian Splatting parameters, all in world space."""

    means: torch.Tensor
    """World-space centers, shape ``(batch, gaussian, 3)``."""

    scales: torch.Tensor
    """Standard-deviation scales, shape ``(batch, gaussian, 3)``."""

    rotations: torch.Tensor
    """World-space orientation as scalar-first (WXYZ) quaternions, shape
    ``(batch, gaussian, 4)``."""

    harmonics: torch.Tensor
    """Spherical-harmonic color coefficients, shape
    ``(batch, gaussian, 3, d_sh)``."""

    opacities: torch.Tensor
    """Per-Gaussian opacity, shape ``(batch, gaussian)``, or its own SH
    coefficients, shape ``(batch, gaussian, 1, d_sh)``."""


@dataclass
class Prediction:
    """A dataclass holding the result of
    :meth:`depth_anything_3.api.DepthAnything3.inference`."""

    depth: np.ndarray
    """Estimated depth maps, shape ``(N, H, W)``."""

    is_metric: int
    """Whether ``depth`` is in real-world units (metric models) rather than an
    arbitrary relative scale."""

    sky: np.ndarray | None = None
    """Sky segmentation mask, shape ``(N, H, W)``, if the model supports it."""

    conf: np.ndarray | None = None
    """Per-pixel confidence, shape ``(N, H, W)``."""

    extrinsics: np.ndarray | None = None
    """World-to-camera matrices, shape ``(N, 4, 4)``."""

    intrinsics: np.ndarray | None = None
    """Camera intrinsic matrices, shape ``(N, 3, 3)``."""

    processed_images: np.ndarray | None = None
    """The resized/normalized input images actually fed to the model, shape
    ``(N, H, W, 3)`` -- useful for visualization alongside ``depth``, since it
    matches its resolution."""

    gaussians: Gaussians | None = None
    """Predicted 3D Gaussians, if ``infer_gs=True`` was passed to
    ``inference()``."""

    aux: dict[str, Any] = None
    """Auxiliary outputs, e.g. ``feat_layer_<i>`` intermediate features when
    ``export_feat_layers`` was set."""

    scale_factor: Optional[float] = None
    """The metric scale factor applied, for metric models."""
