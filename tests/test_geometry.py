"""Rotation and rigid-transform helpers in utils/geometry.py.

These functions carry two conventions that are easy to break silently and
that nothing else in the repo pins down: quaternions are **scalar-last**
(XYZW), and `affine_inverse` assumes the input is a rigid transform.
"""

import numpy as np
import pytest
import torch
from conftest import random_rotations, random_se3

from depth_anything_3.utils.geometry import (
    affine_inverse,
    affine_inverse_np,
    as_homogeneous,
    mat_to_quat,
    quat_to_mat,
    standardize_quaternion,
    transpose_last_two_axes,
)


def test_quat_to_mat_produces_rotations():
    q = torch.nn.functional.normalize(torch.randn(64, 4, dtype=torch.float64), dim=-1)
    m = quat_to_mat(q)
    assert m.shape == (64, 3, 3)
    torch.testing.assert_close(m @ m.mT, torch.eye(3, dtype=torch.float64).expand(64, 3, 3))
    torch.testing.assert_close(torch.det(m), torch.ones(64, dtype=torch.float64))


def test_mat_to_quat_roundtrip():
    """mat -> quat -> mat is exact; quat is only defined up to sign."""
    m = random_rotations(64)
    torch.testing.assert_close(quat_to_mat(mat_to_quat(m)), m)


def test_quat_to_mat_roundtrip_up_to_sign():
    q = torch.nn.functional.normalize(torch.randn(64, 4, dtype=torch.float64), dim=-1)
    back = mat_to_quat(quat_to_mat(q))
    # q and -q are the same rotation, so compare after fixing the sign.
    agree = torch.minimum((back - q).abs().sum(-1), (back + q).abs().sum(-1))
    torch.testing.assert_close(agree, torch.zeros(64, dtype=torch.float64), atol=1e-8, rtol=0)


def test_quaternions_are_scalar_last():
    """A 90 degree rotation about +Z must be (0, 0, sin, cos), not (cos, 0, 0, sin).

    This is the convention the docstrings claim (XYZW / ijkr). If someone
    swaps in a scalar-first implementation, every other test here still
    passes -- this one does not.
    """
    half = torch.tensor(torch.pi / 4, dtype=torch.float64)
    q = torch.tensor([0.0, 0.0, torch.sin(half), torch.cos(half)], dtype=torch.float64)
    expected = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    torch.testing.assert_close(quat_to_mat(q), expected, atol=1e-12, rtol=0)


def test_standardize_quaternion_fixes_sign():
    q = torch.nn.functional.normalize(torch.randn(32, 4, dtype=torch.float64), dim=-1)
    std = standardize_quaternion(q)
    # Same rotation...
    torch.testing.assert_close(quat_to_mat(std), quat_to_mat(q))
    # ...with a canonical sign.
    torch.testing.assert_close(standardize_quaternion(-q), std)


@pytest.mark.parametrize("batch", [(), (5,), (2, 3)])
def test_affine_inverse_matches_linalg_inv(batch):
    n = int(np.prod(batch)) if batch else 1
    a = random_se3(n).reshape(*batch, 4, 4)
    torch.testing.assert_close(affine_inverse(a), torch.linalg.inv(a))


def test_affine_inverse_np_matches_torch():
    a = random_se3(8)
    torch.testing.assert_close(
        torch.from_numpy(affine_inverse_np(a.numpy())),
        affine_inverse(a),
    )


def test_transpose_last_two_axes_matches_mT():
    """The `for np < 2` shim must agree with numpy's own .mT on numpy >= 2."""
    a = np.random.randn(4, 5, 3, 2)
    np.testing.assert_allclose(transpose_last_two_axes(a), a.mT)


def test_transpose_last_two_axes_passes_through_1d():
    a = np.arange(3.0)
    np.testing.assert_array_equal(transpose_last_two_axes(a), a)


def test_as_homogeneous_extends_3x4():
    ext = random_se3(4)[:, :3, :]
    for x in (ext, ext.numpy()):
        out = as_homogeneous(x)
        assert out.shape == (4, 4, 4)
        bottom = np.asarray(out[:, 3, :])
        np.testing.assert_allclose(bottom, np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)))


def test_as_homogeneous_is_identity_on_4x4():
    a = random_se3(3)
    assert as_homogeneous(a) is a


def test_as_homogeneous_rejects_bad_shapes():
    with pytest.raises(ValueError):
        as_homogeneous(torch.zeros(2, 5, 5))
    with pytest.raises(TypeError):
        as_homogeneous([[1, 2], [3, 4]])
