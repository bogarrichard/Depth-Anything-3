"""Recovering cameras from ray maps -- ``utils/ray_utils.py``.

This is the ``--use-ray-pose`` branch: instead of decoding a pose vector, the
model predicts a per-patch ray field and the camera is *solved* out of it by
a hand-rolled weighted RANSAC homography fit followed by a QL decomposition.
Several hundred lines of linear algebra with no library behind it, on a code
path the default settings never take.

The tests build ray fields from a camera that is known exactly and check that
the solver gets that camera back. Nothing here reuses the solver to compute
its own expectations.

Conventions, read off ``camray_to_caminfo``: the image plane is normalised to
``[-1, 1]`` in both axes, ``camray[..., :3]`` is the ray direction and
``camray[..., 3:]`` the ray origin (the camera centre).
"""

import numpy as np
import pytest
import torch
from conftest import random_rotations

from depth_anything_3.utils.ray_utils import (
    camray_to_caminfo,
    find_homography_least_squares_weighted_torch,
    get_extrinsic_from_camray,
    ql_decomposition,
)

NY, NX = 12, 16


def _normalised_plane_grid(ny: int = NY, nx: int = NX) -> torch.Tensor:
    """The same grid ``camray_to_caminfo`` unprojects internally: pixel centres
    of an ``ny x nx`` patch grid, mapped onto ``[-1, 1]`` with a half-cell inset."""
    dx, dy = 1 / nx, 1 / ny
    xs = torch.linspace(-(1 - dx), 1 - dx, nx, dtype=torch.float32)
    ys = torch.linspace(-(1 - dy), 1 - dy, ny, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)  # (ny, nx, 3)


def _camray_from(rotation, focal, principal, centre):
    """Build the ray field a camera with these parameters would produce.

    ``A = R @ L`` with ``L`` lower triangular is exactly the factorisation
    ``ql_decomposition`` inverts, so the solver has to return ``R``, the
    diagonal of ``L`` and its bottom row.
    """
    lower = torch.tensor(
        [[focal[0], 0.0, 0.0], [0.0, focal[1], 0.0], [principal[0], principal[1], 1.0]],
        dtype=torch.float32,
    )
    a = rotation @ lower
    directions = torch.einsum("ij,hwj->hwi", a, _normalised_plane_grid())
    origins = centre.expand(NY, NX, 3)
    return torch.cat([directions, origins], dim=-1)[None, None]  # (1, 1, ny, nx, 6)


# ---------------------------------------------------------------------------
# the building blocks
# ---------------------------------------------------------------------------
def test_ql_decomposition_factors_a_matrix_into_rotation_times_lower_triangular():
    rotation = random_rotations(1, seed=1)[0].float()
    lower = torch.tensor([[1.3, 0, 0], [0, 0.7, 0], [0.2, -0.1, 1.0]])
    q, ell = ql_decomposition(rotation @ lower)

    torch.testing.assert_close(q @ ell, rotation @ lower, atol=1e-5, rtol=0)
    torch.testing.assert_close(q @ q.T, torch.eye(3), atol=1e-5, rtol=0)
    assert torch.all(torch.diag(ell) > 0), "the diagonal is sign-normalised to positive"
    # Lower triangular: nothing above the diagonal.
    assert torch.allclose(torch.triu(ell, diagonal=1), torch.zeros(3, 3), atol=1e-5)


def test_ql_decomposition_recovers_the_factors_it_was_given():
    rotation = random_rotations(1, seed=2)[0].float()
    lower = torch.tensor([[1.3, 0, 0], [0, 0.7, 0], [0.2, -0.1, 1.0]])
    q, ell = ql_decomposition(rotation @ lower)
    torch.testing.assert_close(q, rotation, atol=1e-4, rtol=0)
    torch.testing.assert_close(ell, lower, atol=1e-4, rtol=0)


def test_the_homography_fit_recovers_an_exact_homography():
    src = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.7]])
    h = torch.tensor([[1.2, 0.1, 0.3], [-0.2, 0.9, -0.4], [0.05, -0.02, 1.0]])
    projected = torch.einsum("ij,nj->ni", h, torch.cat([src, torch.ones(5, 1)], dim=-1))
    dst = projected[:, :2] / projected[:, 2:]

    got = find_homography_least_squares_weighted_torch(src, dst, torch.ones(5))

    # Homographies are defined up to scale; normalise before comparing.
    torch.testing.assert_close(got / got[2, 2], h, atol=1e-4, rtol=0)


def test_the_homography_fit_needs_four_correspondences():
    with pytest.raises(ValueError, match="4 points"):
        find_homography_least_squares_weighted_torch(
            torch.zeros(3, 2), torch.zeros(3, 2), torch.ones(3)
        )


def test_the_homography_fit_honours_its_weights():
    """A zero-weighted outlier must not move the solution at all -- that is
    what makes the RANSAC refit above it meaningful."""
    src = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.7]])
    h = torch.tensor([[1.2, 0.1, 0.3], [-0.2, 0.9, -0.4], [0.05, -0.02, 1.0]])
    projected = torch.einsum("ij,nj->ni", h, torch.cat([src, torch.ones(5, 1)], dim=-1))
    dst = projected[:, :2] / projected[:, 2:]

    clean = find_homography_least_squares_weighted_torch(src, dst, torch.ones(5))

    poisoned_src = torch.cat([src, torch.tensor([[9.0, -9.0]])])
    poisoned_dst = torch.cat([dst, torch.tensor([[100.0, 100.0]])])
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    weighted = find_homography_least_squares_weighted_torch(poisoned_src, poisoned_dst, weights)

    torch.testing.assert_close(weighted / weighted[2, 2], clean / clean[2, 2], atol=1e-4, rtol=0)


# ---------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def known_camera():
    rotation = random_rotations(1, seed=11)[0].float()
    focal = (1.25, 0.85)
    principal = (0.06, -0.04)
    centre = torch.tensor([0.4, -1.1, 2.0])
    return rotation, focal, principal, centre


def test_camray_to_caminfo_recovers_the_camera_it_was_built_from(known_camera):
    rotation, focal, principal, centre = known_camera
    camray = _camray_from(rotation, focal, principal, centre)

    r, t, focal_out, principal_out = camray_to_caminfo(camray)

    assert r.shape == (1, 1, 3, 3) and t.shape == (1, 1, 3)
    torch.testing.assert_close(r[0, 0], rotation, atol=1e-3, rtol=0)
    torch.testing.assert_close(t[0, 0], centre, atol=1e-4, rtol=0)
    # The function returns the *inverse* of the fitted diagonal, and shifts the
    # principal point back into [0, 2] -- both documented by da3.py's use of them.
    torch.testing.assert_close(
        focal_out[0, 0],
        torch.tensor([1 / focal[0], 1 / focal[1]]),
        atol=1e-3,
        rtol=0,
    )
    torch.testing.assert_close(
        principal_out[0, 0],
        torch.tensor([principal[0] + 1.0, principal[1] + 1.0]),
        atol=1e-3,
        rtol=0,
    )


def test_the_translation_is_the_confidence_weighted_mean_of_the_ray_origins():
    """Confidence zero on a patch must remove its origin from the average, or
    a single wild ray drags the whole camera with it."""
    rotation = random_rotations(1, seed=12)[0].float()
    camray = _camray_from(rotation, (1.0, 1.0), (0.0, 0.0), torch.tensor([1.0, 2.0, 3.0]))
    camray = camray.clone()
    camray[0, 0, 0, 0, 3:] = torch.tensor([500.0, 500.0, 500.0])

    confidence = torch.ones(1, 1, NY, NX)
    confidence[0, 0, 0, 0] = 0.0

    _, t, _, _ = camray_to_caminfo(camray, confidence=confidence)

    torch.testing.assert_close(t[0, 0], torch.tensor([1.0, 2.0, 3.0]), atol=1e-3, rtol=0)


def test_camray_to_caminfo_is_deterministic(known_camera):
    """It samples inside a RANSAC loop; two runs on the same input must not
    disagree, or the same images would give different cameras."""
    camray = _camray_from(*known_camera)
    first = camray_to_caminfo(camray)
    second = camray_to_caminfo(camray)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)


def test_get_extrinsic_from_camray_assembles_a_homogeneous_matrix(known_camera):
    rotation, focal, principal, centre = known_camera
    camray = _camray_from(rotation, focal, principal, centre)
    conf = torch.ones(1, 1, NY, NX, 1)

    extrinsic, focal_out, principal_out = get_extrinsic_from_camray(camray, conf, NY, NX)

    assert extrinsic.shape == (1, 1, 4, 4)
    torch.testing.assert_close(
        extrinsic[0, 0, 3], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=0, rtol=0
    )
    torch.testing.assert_close(extrinsic[0, 0, :3, :3], rotation, atol=1e-3, rtol=0)
    torch.testing.assert_close(extrinsic[0, 0, :3, 3], centre, atol=1e-4, rtol=0)
    assert focal_out.shape == (1, 1, 2) and principal_out.shape == (1, 1, 2)


def test_several_views_are_solved_independently():
    """The batch path chunks over B*S; two different cameras in one call must
    not bleed into each other."""
    a = random_rotations(1, seed=21)[0].float()
    b = random_rotations(1, seed=22)[0].float()
    first = _camray_from(a, (1.1, 0.9), (0.0, 0.0), torch.tensor([0.0, 0.0, 0.0]))
    second = _camray_from(b, (0.8, 1.4), (0.1, 0.1), torch.tensor([1.0, 2.0, 3.0]))
    camray = torch.cat([first, second], dim=1)  # (1, 2, ny, nx, 6)

    r, t, _, _ = camray_to_caminfo(camray)

    torch.testing.assert_close(r[0, 0], a, atol=1e-3, rtol=0)
    torch.testing.assert_close(r[0, 1], b, atol=1e-3, rtol=0)
    torch.testing.assert_close(t[0, 1], torch.tensor([1.0, 2.0, 3.0]), atol=1e-4, rtol=0)


# ---------------------------------------------------------------------------
# through the model
# ---------------------------------------------------------------------------
def test_use_ray_pose_produces_the_same_output_contract(tiny_net):
    """``--use-ray-pose`` swaps the camera decoder for this solver. It has to
    fill the same keys with the same shapes, or every downstream consumer
    breaks for that flag alone."""
    from conftest import TINY_HW

    h, w = TINY_HW
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 2, 3, h, w, generator=g)

    with torch.no_grad():
        out = tiny_net(x, use_ray_pose=True)

    assert out.extrinsics.shape == (1, 2, 3, 4)
    assert out.intrinsics.shape == (1, 2, 3, 3)
    assert "ray" not in out and "ray_conf" not in out
    assert np.isfinite(out.extrinsics.numpy()).all()
    assert np.isfinite(out.intrinsics.numpy()).all()


def test_ray_pose_intrinsics_are_scaled_into_pixels(tiny_net):
    """``da3.py`` converts the solver's normalised focal/principal values into
    pixel units using the image size. A missing factor there is invisible in
    the shapes and wrong by a factor of two."""
    from conftest import TINY_HW

    h, w = TINY_HW
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 2, 3, h, w, generator=g)

    with torch.no_grad():
        out = tiny_net(x, use_ray_pose=True)

    k = out.intrinsics
    assert torch.all(k[..., 2, 2] == 1.0)
    assert torch.all(k[..., 0, 1] == 0.0) and torch.all(k[..., 1, 0] == 0.0)
    # cx, cy come from `principal * size * 0.5`, so they live on the image.
    assert torch.all(k[..., 0, 2].abs() <= w) and torch.all(k[..., 1, 2].abs() <= h)
