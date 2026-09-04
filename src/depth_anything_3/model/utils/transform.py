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

import torch

from depth_anything_3.utils.geometry import mat_to_quat, quat_to_mat, standardize_quaternion

__all__ = [
    "extri_intri_to_pose_encoding",
    "pose_encoding_to_extri_intri",
    "mat_to_quat",
    "quat_to_mat",
    "standardize_quaternion",
    "cam_quat_xyzw_to_world_quat_wxyz",
]


def extri_intri_to_pose_encoding(
    extrinsics,
    intrinsics,
    image_size_hw=None,
):
    """Convert camera extrinsics and intrinsics to a compact pose encoding."""

    # extrinsics: BxSx3x4
    # intrinsics: BxSx3x3
    R = extrinsics[:, :, :3, :3]  # BxSx3x3
    T = extrinsics[:, :, :3, 3]  # BxSx3

    quat = mat_to_quat(R)
    # Note the order of h and w here
    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
    pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()

    return pose_encoding


def pose_encoding_to_extri_intri(
    pose_encoding,
    image_size_hw=None,
):
    """Convert a pose encoding back to camera extrinsics and intrinsics."""

    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]

    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)

    H, W = image_size_hw
    fy = (H / 2.0) / torch.clamp(torch.tan(fov_h / 2.0), 1e-6)
    fx = (W / 2.0) / torch.clamp(torch.tan(fov_w / 2.0), 1e-6)
    intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = W / 2
    intrinsics[..., 1, 2] = H / 2
    intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1

    return extrinsics, intrinsics


def cam_quat_xyzw_to_world_quat_wxyz(cam_quat_xyzw, c2w):
    # cam_quat_xyzw: (b, n, 4) in xyzw
    # c2w: (b, n, 4, 4)
    b, n = cam_quat_xyzw.shape[:2]
    # 1. xyzw -> wxyz
    cam_quat_wxyz = torch.cat(
        [
            cam_quat_xyzw[..., 3:4],  # w
            cam_quat_xyzw[..., 0:1],  # x
            cam_quat_xyzw[..., 1:2],  # y
            cam_quat_xyzw[..., 2:3],  # z
        ],
        dim=-1,
    )
    # 2. Quaternion to matrix
    cam_quat_wxyz_flat = cam_quat_wxyz.reshape(-1, 4)
    rotmat_cam = quat_to_mat(cam_quat_wxyz_flat).reshape(b, n, 3, 3)
    # 3. Transform to world space
    rotmat_c2w = c2w[..., :3, :3]
    rotmat_world = torch.matmul(rotmat_c2w, rotmat_cam)
    # 4. Matrix to quaternion (wxyz)
    rotmat_world_flat = rotmat_world.reshape(-1, 3, 3)
    world_quat_wxyz_flat = mat_to_quat(rotmat_world_flat)
    world_quat_wxyz = world_quat_wxyz_flat.reshape(b, n, 4)
    return world_quat_wxyz
