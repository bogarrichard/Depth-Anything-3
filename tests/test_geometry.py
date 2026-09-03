"""Rotation and rigid-transform contracts in ``utils/geometry.py``.

Every expectation here comes from outside the package -- scipy, a longhand
Hamilton product, or the block formula for an SE(3) inverse -- so an
implementation that is wrong but self-consistent cannot pass. The two
conventions being pinned are the ones nothing else in the repo states:
quaternions are **scalar-last** (XYZW), and ``affine_inverse`` inverts by
transposing the rotation block, which is only correct for rigid input.
"""

import numpy as np
import pytest
import torch
from _oracles import (
    axis_rotation,
    mat_to_quat_oracle,
    quat_multiply_oracle,
    quat_to_mat_oracle,
    se3_inverse_oracle,
)
from conftest import random_rotations, random_se3, random_unit_quaternions

from depth_anything_3.utils.geometry import (
    affine_inverse,
    affine_inverse_np,
    as_homogeneous,
    mat_to_quat,
    quat_to_mat,
    so3_to_mat,
    standardize_quaternion,
    transpose_last_two_axes,
)


def _quat_agree(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance between quaternions treating q and -q as equal."""
    return torch.minimum((a - b).abs().amax(-1), (a + b).abs().amax(-1))


# ---------------------------------------------------------------------------
# quat_to_mat
# ---------------------------------------------------------------------------
def test_quat_to_mat_matches_scipy():
    """The independent check: scipy also reads quaternions scalar-last."""
    q = random_unit_quaternions(64)
    expected = torch.from_numpy(quat_to_mat_oracle(q.numpy()))
    torch.testing.assert_close(quat_to_mat(q), expected, atol=1e-12, rtol=0)


def test_quat_to_mat_is_scalar_last():
    """A +90 degree turn about +Z is (0, 0, sin, cos), not (cos, 0, 0, sin).

    Spelled out rather than delegated, because this single fact decides
    whether every stored camera rotation in the repo is read correctly. A
    scalar-first implementation passes every round trip in this file and
    fails only here.
    """
    half = torch.tensor(torch.pi / 4, dtype=torch.float64)
    q = torch.tensor([0.0, 0.0, torch.sin(half), torch.cos(half)], dtype=torch.float64)
    torch.testing.assert_close(
        quat_to_mat(q),
        torch.from_numpy(axis_rotation("z", np.pi / 2)),
        atol=1e-12,
        rtol=0,
    )


def test_quat_to_mat_output_is_a_rotation():
    m = quat_to_mat(random_unit_quaternions(64))
    eye = torch.eye(3, dtype=torch.float64).expand(64, 3, 3)
    torch.testing.assert_close(m @ m.mT, eye)
    torch.testing.assert_close(torch.det(m), torch.ones(64, dtype=torch.float64))


def test_quat_to_mat_ignores_quaternion_magnitude():
    """The ``2 / |q|^2`` factor makes the map scale invariant; callers rely on
    it to skip normalising network output."""
    q = random_unit_quaternions(32)
    scales = torch.linspace(0.25, 4.0, 32, dtype=torch.float64)[:, None]
    torch.testing.assert_close(quat_to_mat(q * scales), quat_to_mat(q))


def test_quat_to_mat_is_a_homomorphism():
    """R(p * q) == R(p) @ R(q), with the product computed longhand.

    This is the property that fixes the handedness and the multiplication
    order together; neither a transposed nor a conjugated implementation
    survives it.
    """
    p = random_unit_quaternions(48, seed=1)
    q = random_unit_quaternions(48, seed=2)
    product = torch.from_numpy(quat_multiply_oracle(p.numpy(), q.numpy()))
    torch.testing.assert_close(quat_to_mat(product), quat_to_mat(p) @ quat_to_mat(q))


@pytest.mark.parametrize("batch", [(), (5,), (2, 3)])
def test_quat_to_mat_preserves_batch_shape(batch):
    n = int(np.prod(batch)) if batch else 1
    q = random_unit_quaternions(n).reshape(*batch, 4)
    assert quat_to_mat(q).shape == (*batch, 3, 3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quat_to_mat_preserves_dtype(dtype):
    assert quat_to_mat(random_unit_quaternions(4).to(dtype)).dtype == dtype


# ---------------------------------------------------------------------------
# mat_to_quat
# ---------------------------------------------------------------------------
def test_mat_to_quat_matches_scipy():
    m = random_rotations(64)
    expected = torch.from_numpy(mat_to_quat_oracle(m.numpy()))
    torch.testing.assert_close(
        _quat_agree(mat_to_quat(m), expected),
        torch.zeros(64, dtype=torch.float64),
        atol=1e-8,
        rtol=0,
    )


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_mat_to_quat_handles_180_degree_rotations(axis):
    """The branch-selection cases: at 180 degrees the real part vanishes and
    three of the four candidate quaternions are ill-conditioned. Random
    rotations essentially never land here, so it has to be asked for."""
    m = torch.from_numpy(axis_rotation(axis, np.pi))[None]
    expected = torch.from_numpy(mat_to_quat_oracle(m.numpy()))
    torch.testing.assert_close(
        _quat_agree(mat_to_quat(m), expected),
        torch.zeros(1, dtype=torch.float64),
        atol=1e-7,
        rtol=0,
    )


def test_mat_to_quat_of_identity_is_the_identity_quaternion():
    m = torch.eye(3, dtype=torch.float64)[None]
    torch.testing.assert_close(
        mat_to_quat(m), torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    )


def test_mat_to_quat_output_is_standardized():
    """Documented post-condition: the returned real part is never negative."""
    q = mat_to_quat(random_rotations(128))
    assert torch.all(q[..., 3] >= 0)


def test_mat_to_quat_roundtrips_through_quat_to_mat():
    m = random_rotations(64)
    torch.testing.assert_close(quat_to_mat(mat_to_quat(m)), m)


def test_mat_to_quat_rejects_non_3x3_input():
    with pytest.raises(ValueError):
        mat_to_quat(torch.zeros(4, 3, 4, dtype=torch.float64))


def test_mat_to_quat_is_differentiable():
    """``_sqrt_positive_part`` takes a *different* branch when grad is
    enabled, so the gradient path is genuinely separate code."""
    m = random_rotations(8).clone().requires_grad_(True)
    mat_to_quat(m).sum().backward()
    assert m.grad is not None and torch.isfinite(m.grad).all()


# ---------------------------------------------------------------------------
# standardize_quaternion
# ---------------------------------------------------------------------------
def test_standardize_quaternion_preserves_the_rotation():
    q = random_unit_quaternions(32)
    torch.testing.assert_close(quat_to_mat(standardize_quaternion(q)), quat_to_mat(q))


def test_standardize_quaternion_is_sign_canonical_and_idempotent():
    q = random_unit_quaternions(32)
    std = standardize_quaternion(q)
    assert torch.all(std[..., 3] >= 0)
    torch.testing.assert_close(standardize_quaternion(-q), std)
    torch.testing.assert_close(standardize_quaternion(std), std)


# ---------------------------------------------------------------------------
# affine_inverse
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch", [(), (5,), (2, 3)])
def test_affine_inverse_matches_the_block_formula(batch):
    n = int(np.prod(batch)) if batch else 1
    a = random_se3(n).reshape(*batch, 4, 4)
    torch.testing.assert_close(affine_inverse(a), torch.from_numpy(se3_inverse_oracle(a.numpy())))


def test_affine_inverse_composes_to_the_identity():
    a = random_se3(16)
    torch.testing.assert_close(
        affine_inverse(a) @ a, torch.eye(4, dtype=torch.float64).expand_as(a)
    )


def test_affine_inverse_accepts_3x4_extrinsics():
    """Load-bearing: ``pose_encoding_to_extri_intri`` returns (..., 3, 4) and
    ``DepthAnything3Net._process_camera_estimation`` feeds that straight in.
    The 4x4 bottom row is simply absent, and must not be fabricated."""
    a = random_se3(4)
    out = affine_inverse(a[:, :3, :])
    assert out.shape == (4, 3, 4)
    torch.testing.assert_close(out, affine_inverse(a)[:, :3, :])


def test_affine_inverse_assumes_rigid_input():
    """It transposes rather than inverts, so a scaled rotation is *not*
    inverted correctly. Swapping in ``torch.linalg.inv`` would change results
    wherever a non-rigid matrix reaches it -- this records that boundary."""
    a = random_se3(4).clone()
    a[:, :3, :3] *= 2.0
    assert not torch.allclose(affine_inverse(a), torch.linalg.inv(a))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_affine_inverse_preserves_dtype(dtype):
    assert affine_inverse(random_se3(2).to(dtype)).dtype == dtype


def test_affine_inverse_np_matches_the_torch_version():
    a = random_se3(8)
    torch.testing.assert_close(torch.from_numpy(affine_inverse_np(a.numpy())), affine_inverse(a))


# ---------------------------------------------------------------------------
# array plumbing
# ---------------------------------------------------------------------------
def test_transpose_last_two_axes_matches_numpys_mT():
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
        np.testing.assert_allclose(np.asarray(out[:, :3, :]), np.asarray(x))
        np.testing.assert_allclose(np.asarray(out[:, 3, :]), np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)))


def test_as_homogeneous_is_identity_on_4x4():
    a = random_se3(3)
    assert as_homogeneous(a) is a


def test_as_homogeneous_rejects_bad_shapes():
    with pytest.raises(ValueError):
        as_homogeneous(torch.zeros(2, 5, 5))
    with pytest.raises(ValueError):
        as_homogeneous(np.zeros((2, 5, 5)))
    with pytest.raises(TypeError):
        as_homogeneous([[1, 2], [3, 4]])


# ---------------------------------------------------------------------------
# so3_to_mat: the pypose bridge
# ---------------------------------------------------------------------------
def test_so3_to_mat_accepts_a_raw_tensor():
    q = random_unit_quaternions(32)
    torch.testing.assert_close(so3_to_mat(q), quat_to_mat(q))


def test_so3_to_mat_accepts_a_pypose_so3():
    """The point of the helper: pypose types in, closed-form matrix out."""
    pp = pytest.importorskip("pypose")
    q = random_unit_quaternions(32)
    # Bit-identical, not merely close -- it is the same code path on the same data.
    assert torch.equal(so3_to_mat(pp.SO3(q)), quat_to_mat(q))


def test_so3_to_mat_agrees_with_pypose_matrix():
    """Guards the shortcut: pypose's own ``SO3.matrix()`` is deliberately not
    called, so a pypose convention change has to be caught here."""
    pp = pytest.importorskip("pypose")
    q = random_unit_quaternions(32)
    torch.testing.assert_close(so3_to_mat(pp.SO3(q)), pp.SO3(q).matrix().to(torch.float64))


def test_so3_to_mat_agrees_with_scipy():
    """...and that pypose convention is itself the scalar-last one."""
    pp = pytest.importorskip("pypose")
    q = random_unit_quaternions(32)
    expected = torch.from_numpy(quat_to_mat_oracle(q.numpy()))
    torch.testing.assert_close(so3_to_mat(pp.SO3(q)), expected, atol=1e-12, rtol=0)
