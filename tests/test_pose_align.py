"""Umeyama Sim(3) trajectory alignment in ``utils/pose_align.py``.

Extrinsics here are world-to-camera; ``pose`` means the camera-to-world
inverse. The alignment maps *estimated* poses onto *reference* poses.

Two things this file is careful about:

* the fit is checked against a from-scratch Umeyama implementation
  (``tests/_oracles.py``), not only against a round trip -- the module was
  recently reimplemented on top of ``pypose.svdstf``, and "the new one agrees
  with itself" would have said nothing about that swap;
* the RANSAC path is exercised **with outliers in the data**. On clean input
  RANSAC and plain least squares agree by construction, so a clean-data test
  passes even if the robust path is completely broken.
"""

import numpy as np
import pytest
from _oracles import umeyama_sim3_oracle
from conftest import random_rotations, random_se3

from depth_anything_3.utils.geometry import affine_inverse_np
from depth_anything_3.utils.pose_align import (
    align_poses_umeyama,
    apply_umeyama_alignment_to_ext,
    batch_align_poses_umeyama,
    transform_points_sim3,
)

N = 12


def _make_pair(scale=2.5, seed=0, n=N):
    """Build a reference trajectory and an estimate related to it by a known Sim(3).

    Returns (ext_ref, ext_est, r0, t0, s0) with extrinsics as (n, 4, 4).
    """
    pose_ref = random_se3(n, seed=seed).numpy()
    r0 = random_rotations(1, seed=seed + 7)[0].numpy()
    t0 = np.array([0.3, -1.2, 0.75])
    s0 = scale

    # pose_est is pose_ref pushed through the inverse Sim(3), so that applying
    # (r0, t0, s0) to pose_est recovers pose_ref exactly.
    pose_est = pose_ref.copy()
    pose_est[:, :3, :3] = r0.T @ pose_ref[:, :3, :3]
    pose_est[:, :3, 3] = ((pose_ref[:, :3, 3] - t0) @ r0) / s0

    return affine_inverse_np(pose_ref), affine_inverse_np(pose_est), r0, t0, s0


def _centres(ext):
    return affine_inverse_np(ext)[:, :3, 3]


# ---------------------------------------------------------------------------
# the plain fit
# ---------------------------------------------------------------------------
def test_recovers_a_known_sim3():
    ext_ref, ext_est, r0, t0, s0 = _make_pair()
    r, t, s = align_poses_umeyama(ext_ref, ext_est)
    np.testing.assert_allclose(r, r0, atol=1e-8)
    np.testing.assert_allclose(t, t0, atol=1e-8)
    np.testing.assert_allclose(s, s0, atol=1e-8)


def test_matches_a_from_scratch_umeyama_fit_on_noisy_data():
    """The independent judge. Noise is added so the fit is a genuine least
    squares problem rather than an exactly-determined one."""
    rng = np.random.default_rng(0)
    ext_ref, ext_est, *_ = _make_pair(n=25)
    pose_ref = affine_inverse_np(ext_ref)
    pose_ref[:, :3, 3] += rng.normal(scale=0.05, size=(25, 3))
    ext_ref = affine_inverse_np(pose_ref)

    r, t, s = align_poses_umeyama(ext_ref, ext_est)
    r_ref, t_ref, s_ref = umeyama_sim3_oracle(_centres(ext_est), _centres(ext_ref))

    np.testing.assert_allclose(r, r_ref, atol=1e-8)
    np.testing.assert_allclose(t, t_ref, atol=1e-8)
    np.testing.assert_allclose(s, s_ref, atol=1e-8)


def test_the_fit_is_over_camera_centres_not_extrinsic_translations():
    """``pose_ref ~= s r pose_est + t`` is stated in terms of camera centres.
    Fitting the raw ``ext`` translation columns instead would give a
    different -- and wrong -- answer, so the two must not coincide here."""
    ext_ref, ext_est, *_ = _make_pair()
    r, t, s = align_poses_umeyama(ext_ref, ext_est)
    wrong_r, wrong_t, wrong_s = umeyama_sim3_oracle(ext_est[:, :3, 3], ext_ref[:, :3, 3])
    assert not np.allclose(r, wrong_r, atol=1e-3)
    assert not np.allclose(t, wrong_t, atol=1e-3) or not np.isclose(s, wrong_s, atol=1e-3)


def test_aligned_extrinsics_match_the_reference():
    ext_ref, ext_est, *_ = _make_pair()
    *_, ext_aligned = align_poses_umeyama(ext_ref, ext_est, return_aligned=True)
    np.testing.assert_allclose(ext_aligned, ext_ref, atol=1e-8)


def test_identity_alignment_is_a_noop():
    ext = affine_inverse_np(random_se3(N).numpy())
    r, t, s = align_poses_umeyama(ext, ext)
    np.testing.assert_allclose(r, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(t, np.zeros(3), atol=1e-8)
    np.testing.assert_allclose(s, 1.0, atol=1e-8)


def test_accepts_3x4_extrinsics():
    """_to44 should widen (N, 3, 4) input to (N, 4, 4)."""
    ext_ref, ext_est, r0, t0, s0 = _make_pair()
    r, t, s = align_poses_umeyama(ext_ref[:, :3, :], ext_est[:, :3, :])
    np.testing.assert_allclose(r, r0, atol=1e-8)
    np.testing.assert_allclose(t, t0, atol=1e-8)
    np.testing.assert_allclose(s, s0, atol=1e-8)


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_scale_is_recovered_across_orders_of_magnitude(scale):
    ext_ref, ext_est, _, _, s0 = _make_pair(scale=scale)
    _, _, s = align_poses_umeyama(ext_ref, ext_est)
    np.testing.assert_allclose(s, s0, rtol=1e-6)


# ---------------------------------------------------------------------------
# the RANSAC path -- judged on data it is actually needed for
# ---------------------------------------------------------------------------
def _with_outliers(n=20, n_outliers=2, seed=0):
    ext_ref, ext_est, r0, t0, s0 = _make_pair(n=n, seed=seed)
    pose_est = affine_inverse_np(ext_est)
    rng = np.random.default_rng(seed + 100)
    corrupted = rng.choice(n, size=n_outliers, replace=False)
    pose_est[corrupted, :3, 3] += rng.normal(scale=20.0, size=(n_outliers, 3))
    return ext_ref, affine_inverse_np(pose_est), r0, t0, s0


def _sim3_error(r, t, s, r0, t0, s0):
    return max(float(np.abs(r - r0).max()), float(np.abs(t - t0).max()), abs(float(s) - s0) / s0)


def test_ransac_beats_plain_least_squares_when_the_data_has_outliers():
    """The whole reason the branch exists. On clean data both paths agree, so
    this is the only test that can tell a working RANSAC from a broken one."""
    ext_ref, ext_est, r0, t0, s0 = _with_outliers()

    plain = align_poses_umeyama(ext_ref, ext_est)
    robust = align_poses_umeyama(ext_ref, ext_est, ransac=True, random_state=0)

    plain_error = _sim3_error(*plain, r0, t0, s0)
    robust_error = _sim3_error(*robust, r0, t0, s0)
    assert robust_error < 1e-6, f"RANSAC did not recover the true transform ({robust_error})"
    assert robust_error < plain_error / 10


def test_ransac_still_recovers_a_clean_transform():
    ext_ref, ext_est, r0, t0, s0 = _make_pair()
    r, t, s = align_poses_umeyama(ext_ref, ext_est, ransac=True, random_state=0)
    np.testing.assert_allclose(r, r0, atol=1e-6)
    np.testing.assert_allclose(t, t0, atol=1e-6)
    np.testing.assert_allclose(s, s0, atol=1e-6)


def test_ransac_is_reproducible_for_a_fixed_random_state():
    """``api.py`` pins ``random_state=42``; without reproducibility the same
    images would give different poses run to run."""
    ext_ref, ext_est, *_ = _with_outliers()
    first = align_poses_umeyama(ext_ref, ext_est, ransac=True, random_state=7)
    second = align_poses_umeyama(ext_ref, ext_est, ransac=True, random_state=7)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# applying an alignment
# ---------------------------------------------------------------------------
def test_apply_alignment_reproduces_return_aligned():
    """Two separate code paths compute the same thing -- ``api.py`` uses the
    ``return_aligned`` output, other callers re-apply the transform later."""
    ext_ref, ext_est, *_ = _make_pair()
    r, t, s, aligned = align_poses_umeyama(ext_ref, ext_est, return_aligned=True)
    np.testing.assert_allclose(
        apply_umeyama_alignment_to_ext(r, t, s, ext_est), aligned, atol=1e-9
    )


def test_apply_alignment_accepts_3x4_and_widens_it():
    ext_ref, ext_est, *_ = _make_pair()
    r, t, s, aligned = align_poses_umeyama(ext_ref, ext_est, return_aligned=True)
    out = apply_umeyama_alignment_to_ext(r, t, s, ext_est[:, :3, :])
    assert out.shape == (N, 4, 4)
    np.testing.assert_allclose(out, aligned, atol=1e-9)


def test_batch_alignment_matches_the_single_trajectory_fit():
    """``gs_adapter`` calls the batched torch wrapper; it must agree with the
    numpy function it delegates to, including for a batch of size > 1."""
    import torch

    pairs = [_make_pair(seed=0), _make_pair(seed=3, scale=0.4)]
    ext_ref = torch.from_numpy(np.stack([p[0] for p in pairs]))
    ext_est = torch.from_numpy(np.stack([p[1] for p in pairs]))

    rots, trans, scales = batch_align_poses_umeyama(ext_ref, ext_est)

    assert rots.shape == (2, 3, 3) and trans.shape == (2, 3) and scales.shape == (2,)
    for i, (_, _, r0, t0, s0) in enumerate(pairs):
        np.testing.assert_allclose(rots[i].numpy(), r0, atol=1e-8)
        np.testing.assert_allclose(trans[i].numpy(), t0, atol=1e-8)
        np.testing.assert_allclose(float(scales[i]), s0, atol=1e-8)


def test_batch_alignment_refuses_tensors_that_require_grad():
    """It drops to numpy internally, so a graph would be silently severed."""
    import torch

    ext_ref, ext_est, *_ = _make_pair()
    a = torch.from_numpy(ext_ref)[None].requires_grad_(True)
    b = torch.from_numpy(ext_est)[None]
    with pytest.raises(AssertionError):
        batch_align_poses_umeyama(a, b)


# ---------------------------------------------------------------------------
# point transforms
# ---------------------------------------------------------------------------
def test_transform_points_sim3_matches_the_formula():
    pts = np.random.randn(50, 3)
    rot = random_rotations(1, seed=3)[0].numpy()
    trans = np.array([1.0, -2.0, 0.5])
    expected = 3.0 * (rot @ pts.T).T + trans
    np.testing.assert_allclose(transform_points_sim3(pts, rot, trans, 3.0), expected, atol=1e-12)


def test_transform_points_sim3_roundtrips():
    pts = np.random.randn(50, 3)
    rot = random_rotations(1, seed=3)[0].numpy()
    trans = np.array([1.0, -2.0, 0.5])
    fwd = transform_points_sim3(pts, rot, trans, 3.0)
    back = transform_points_sim3(fwd, rot, trans, 3.0, inverse=True)
    np.testing.assert_allclose(back, pts, atol=1e-10)


def test_transform_points_sim3_agrees_with_the_pose_alignment_it_names():
    """The point transform and the pose transform have to be the same Sim(3);
    otherwise a re-scaled point cloud stops matching its own cameras."""
    ext_ref, ext_est, *_ = _make_pair()
    r, t, s = align_poses_umeyama(ext_ref, ext_est)
    centres_est = _centres(ext_est)
    np.testing.assert_allclose(
        transform_points_sim3(centres_est, r, t, s), _centres(ext_ref), atol=1e-8
    )
