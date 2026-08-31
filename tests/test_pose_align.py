"""Umeyama Sim(3) trajectory alignment in utils/pose_align.py.

Extrinsics here are world-to-camera; `pose` means the camera-to-world
inverse. The alignment maps *estimated* poses onto *reference* poses.
"""

import numpy as np
from conftest import random_rotations, random_se3

from depth_anything_3.utils.geometry import affine_inverse_np
from depth_anything_3.utils.pose_align import align_poses_umeyama, transform_points_sim3

N = 12


def _make_pair(scale=2.5, seed=0):
    """Build a reference trajectory and an estimate related to it by a known Sim(3).

    Returns (ext_ref, ext_est, r0, t0, s0) with extrinsics as (N, 4, 4).
    """
    pose_ref = random_se3(N, seed=seed).numpy()
    r0 = random_rotations(1, seed=seed + 7)[0].numpy()
    t0 = np.array([0.3, -1.2, 0.75])
    s0 = scale

    # pose_est is pose_ref pushed through the inverse Sim(3), so that applying
    # (r0, t0, s0) to pose_est recovers pose_ref exactly.
    pose_est = pose_ref.copy()
    pose_est[:, :3, :3] = r0.T @ pose_ref[:, :3, :3]
    pose_est[:, :3, 3] = ((pose_ref[:, :3, 3] - t0) @ r0) / s0

    return affine_inverse_np(pose_ref), affine_inverse_np(pose_est), r0, t0, s0


def test_recovers_a_known_sim3():
    ext_ref, ext_est, r0, t0, s0 = _make_pair()
    r, t, s = align_poses_umeyama(ext_ref, ext_est)
    np.testing.assert_allclose(r, r0, atol=1e-8)
    np.testing.assert_allclose(t, t0, atol=1e-8)
    np.testing.assert_allclose(s, s0, atol=1e-8)


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
    np.testing.assert_allclose(s, s0, atol=1e-8)


def test_ransac_path_recovers_the_same_transform():
    ext_ref, ext_est, r0, t0, s0 = _make_pair()
    r, t, s = align_poses_umeyama(ext_ref, ext_est, ransac=True, random_state=0)
    np.testing.assert_allclose(r, r0, atol=1e-6)
    np.testing.assert_allclose(t, t0, atol=1e-6)
    np.testing.assert_allclose(s, s0, atol=1e-6)


def test_transform_points_sim3_roundtrips():
    pts = np.random.randn(50, 3)
    rot = random_rotations(1, seed=3)[0].numpy()
    trans = np.array([1.0, -2.0, 0.5])
    fwd = transform_points_sim3(pts, rot, trans, 3.0)
    back = transform_points_sim3(fwd, rot, trans, 3.0, inverse=True)
    np.testing.assert_allclose(back, pts, atol=1e-10)
