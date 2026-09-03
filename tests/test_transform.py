"""``model/utils/transform.py``: the pose encoding the camera head emits.

Two separate contracts live in this module and both are load-bearing for
pretrained checkpoints:

* the **pose encoding layout** -- 9 numbers, ``[t(3) | quat_xyzw(4) | fov_h |
  fov_w]`` -- which is what ``CameraDec`` was trained to produce, and
* the **quaternion helpers**, which are a near-verbatim second copy of the
  ones in ``utils/geometry.py``.

The copies are checked against scipy *individually* rather than against each
other: "the two agree" is satisfied by two identically wrong functions. The
pairwise comparison is kept as well, but only as the safety net for merging
them (backlog item 8) -- do not delete it before that refactor.
"""

import numpy as np
import pytest
import torch
from _oracles import mat_to_quat_oracle, quat_to_mat_oracle
from conftest import random_rotations, random_se3, random_unit_quaternions

from depth_anything_3.model.utils import transform as t
from depth_anything_3.utils import geometry as g

B, N = 2, 3
IMAGE_HW = (56, 70)


# ---------------------------------------------------------------------------
# the duplicated quaternion helpers, judged independently
# ---------------------------------------------------------------------------
def test_transform_quat_to_mat_matches_scipy():
    q = random_unit_quaternions(64)
    torch.testing.assert_close(
        t.quat_to_mat(q), torch.from_numpy(quat_to_mat_oracle(q.numpy())), atol=1e-12, rtol=0
    )


def test_transform_mat_to_quat_matches_scipy():
    m = random_rotations(64)
    got = t.mat_to_quat(m)
    expected = torch.from_numpy(mat_to_quat_oracle(m.numpy()))
    agree = torch.minimum((got - expected).abs().amax(-1), (got + expected).abs().amax(-1))
    torch.testing.assert_close(agree, torch.zeros(64, dtype=torch.float64), atol=1e-8, rtol=0)


def test_transform_standardize_quaternion_is_sign_canonical():
    q = random_unit_quaternions(32)
    assert torch.all(t.standardize_quaternion(q)[..., 3] >= 0)
    torch.testing.assert_close(t.standardize_quaternion(-q), t.standardize_quaternion(q))


@pytest.mark.parametrize("name", ["quat_to_mat", "mat_to_quat", "standardize_quaternion"])
def test_the_two_copies_still_agree(name):
    """Safety net for merging the modules -- see backlog item 8."""
    arg = random_rotations(64) if name == "mat_to_quat" else random_unit_quaternions(64)
    torch.testing.assert_close(getattr(g, name)(arg), getattr(t, name)(arg))


# ---------------------------------------------------------------------------
# pose encoding
# ---------------------------------------------------------------------------
def _extrinsics_and_intrinsics():
    ext = random_se3(B * N).reshape(B, N, 4, 4)[..., :3, :]
    h, w = IMAGE_HW
    k = torch.zeros(B, N, 3, 3, dtype=torch.float64)
    k[..., 0, 0] = 40.0
    k[..., 1, 1] = 30.0
    k[..., 0, 2] = w / 2
    k[..., 1, 2] = h / 2
    k[..., 2, 2] = 1.0
    return ext, k


def test_pose_encoding_layout_is_translation_quaternion_then_two_fovs():
    """Pinning the field order: a checkpoint's camera head emits exactly this
    vector, so reordering it silently invalidates every pretrained model."""
    ext, k = _extrinsics_and_intrinsics()
    h, w = IMAGE_HW

    enc = t.extri_intri_to_pose_encoding(ext, k, image_size_hw=IMAGE_HW)

    assert enc.shape == (B, N, 9)
    torch.testing.assert_close(enc[..., :3], ext[..., :3, 3].float())
    torch.testing.assert_close(enc[..., 3:7], t.mat_to_quat(ext[..., :3, :3]).float())
    torch.testing.assert_close(
        enc[..., 7], (2 * torch.atan((h / 2) / k[..., 1, 1])).float(), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        enc[..., 8], (2 * torch.atan((w / 2) / k[..., 0, 0])).float(), atol=1e-6, rtol=0
    )


def test_pose_encoding_roundtrips_through_extrinsics_and_intrinsics():
    ext, k = _extrinsics_and_intrinsics()
    enc = t.extri_intri_to_pose_encoding(ext, k, image_size_hw=IMAGE_HW)
    ext_back, k_back = t.pose_encoding_to_extri_intri(enc, image_size_hw=IMAGE_HW)

    assert ext_back.shape == (B, N, 3, 4)
    torch.testing.assert_close(ext_back, ext.float(), atol=1e-5, rtol=0)
    torch.testing.assert_close(k_back, k.float(), atol=1e-3, rtol=0)


def test_decoded_intrinsics_are_a_centred_pinhole():
    """The decoder cannot recover a principal point -- it always re-centres."""
    h, w = IMAGE_HW
    enc = torch.zeros(1, 1, 9, dtype=torch.float32)
    enc[..., 6] = 1.0  # unit quaternion, scalar-last
    enc[..., 7] = 2 * np.arctan((h / 2) / 30.0)
    enc[..., 8] = 2 * np.arctan((w / 2) / 40.0)

    _, k = t.pose_encoding_to_extri_intri(enc, image_size_hw=IMAGE_HW)

    torch.testing.assert_close(k[0, 0, 0, 0], torch.tensor(40.0), atol=1e-3, rtol=0)
    torch.testing.assert_close(k[0, 0, 1, 1], torch.tensor(30.0), atol=1e-3, rtol=0)
    assert k[0, 0, 0, 2] == w / 2 and k[0, 0, 1, 2] == h / 2
    assert k[0, 0, 2, 2] == 1.0
    assert k[0, 0, 0, 1] == 0.0 and k[0, 0, 1, 0] == 0.0


def test_decoded_focal_length_is_clamped_for_a_degenerate_fov():
    """``tan(fov/2)`` is clamped at 1e-6, so a zero field of view yields a
    huge but finite focal length instead of an inf."""
    enc = torch.zeros(1, 1, 9, dtype=torch.float32)
    enc[..., 6] = 1.0
    _, k = t.pose_encoding_to_extri_intri(enc, image_size_hw=IMAGE_HW)
    assert torch.isfinite(k).all()
    assert k[0, 0, 0, 0] > 1e6


# ---------------------------------------------------------------------------
# camera-space -> world-space gaussian rotations
# ---------------------------------------------------------------------------
def _roll_to_wxyz(q_xyzw: torch.Tensor) -> torch.Tensor:
    """The permutation ``cam_quat_xyzw_to_world_quat_wxyz`` applies on input."""
    return torch.cat([q_xyzw[..., 3:4], q_xyzw[..., 0:3]], dim=-1)


def test_world_quaternion_with_an_identity_pose_is_the_permuted_input():
    """Pins the exact channel permutation the function applies.

    The network's raw rotation output is arbitrary, so the permutation is
    only meaningful as a *fixed* convention that trained weights encode. Any
    change here -- including "fixing" it -- silently rotates every gaussian.
    """
    q = random_unit_quaternions(B * N).reshape(B, N, 4)
    c2w = torch.eye(4, dtype=torch.float64).expand(B, N, 4, 4)

    out = t.cam_quat_xyzw_to_world_quat_wxyz(q, c2w)

    torch.testing.assert_close(out, t.standardize_quaternion(_roll_to_wxyz(q)))


def test_world_quaternion_is_a_left_rotation_by_the_camera_pose():
    """Whatever the channel names say, the value returned decodes as
    scalar-last and equals ``R_c2w @ R(permuted input)``."""
    q = random_unit_quaternions(B * N).reshape(B, N, 4)
    c2w = random_se3(B * N).reshape(B, N, 4, 4)

    out = t.cam_quat_xyzw_to_world_quat_wxyz(q, c2w)

    expected = c2w[..., :3, :3] @ t.quat_to_mat(_roll_to_wxyz(q))
    torch.testing.assert_close(t.quat_to_mat(out), expected)


def test_world_quaternion_is_a_unit_quaternion():
    q = random_unit_quaternions(B * N).reshape(B, N, 4)
    c2w = random_se3(B * N).reshape(B, N, 4, 4)
    out = t.cam_quat_xyzw_to_world_quat_wxyz(q, c2w)
    assert out.shape == (B, N, 4)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(B, N, dtype=torch.float64))
