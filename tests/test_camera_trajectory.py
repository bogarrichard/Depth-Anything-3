"""Camera-path helpers in ``utils/camera_trj_helpers.py``.

``matrix_to_euler``/``euler_to_matrix`` delegate to scipy, so a round trip
tests nothing about them -- scipy is self-consistent by construction. What
can break, and silently, is the *convention*: scipy reads a lowercase pattern
as extrinsic (fixed-axis) and an uppercase one as intrinsic (body-fixed).
Those are pinned here against explicit axis-rotation products, so a scipy
change or a pattern typo is caught rather than absorbed.

``render_stabilization_path`` is checked on a signal whose smoothed value is
known exactly: a constant trajectory must survive smoothing untouched.
"""

import numpy as np
import pytest
import torch
from _oracles import axis_rotation
from conftest import random_rotations, random_se3

from depth_anything_3.utils.camera_trj_helpers import (
    euler_to_matrix,
    generate_coordinate_frame,
    interpolate_intrinsics,
    intersect_rays,
    matrix_to_euler,
    render_dolly_zoom_path,
    render_stabilization_path,
    render_wander_path,
)

PATTERNS = ["xyz", "zyx", "ZYX"]


# ---------------------------------------------------------------------------
# Euler conventions
# ---------------------------------------------------------------------------
def test_lowercase_pattern_is_extrinsic():
    """``"xyz"`` means: rotate about the *fixed* x, then y, then z axis, so
    the matrices multiply right-to-left."""
    a, b, c = 0.3, -0.7, 1.1
    got = euler_to_matrix(torch.tensor([[a, b, c]], dtype=torch.float64), "xyz")
    expected = axis_rotation("z", c) @ axis_rotation("y", b) @ axis_rotation("x", a)
    torch.testing.assert_close(got[0], torch.from_numpy(expected), atol=1e-12, rtol=0)


def test_uppercase_pattern_is_intrinsic():
    """``"XYZ"`` rotates about the axes carried along by the body, which
    reverses the multiplication order."""
    a, b, c = 0.3, -0.7, 1.1
    got = euler_to_matrix(torch.tensor([[a, b, c]], dtype=torch.float64), "XYZ")
    expected = axis_rotation("x", a) @ axis_rotation("y", b) @ axis_rotation("z", c)
    torch.testing.assert_close(got[0], torch.from_numpy(expected), atol=1e-12, rtol=0)


def test_the_two_conventions_really_differ():
    """Guard the guard: if scipy ever stopped distinguishing case, both tests
    above would still pass while meaning nothing."""
    angles = torch.tensor([[0.3, -0.7, 1.1]], dtype=torch.float64)
    assert not torch.allclose(euler_to_matrix(angles, "xyz"), euler_to_matrix(angles, "XYZ"))


@pytest.mark.parametrize("pattern", PATTERNS)
def test_matrix_euler_roundtrip(pattern):
    m = random_rotations(32)
    torch.testing.assert_close(euler_to_matrix(matrix_to_euler(m, pattern), pattern), m)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_euler_matrix_roundtrip_recovers_the_matrix(pattern):
    """Angles are not unique, so compare the matrices rather than the angles."""
    angles = torch.rand(32, 3, dtype=torch.float64) * 2 - 1  # within the principal range
    m = euler_to_matrix(angles, pattern)
    torch.testing.assert_close(euler_to_matrix(matrix_to_euler(m, pattern), pattern), m)


@pytest.mark.parametrize("batch", [(4,), (2, 3)])
def test_batch_shapes_are_preserved(batch):
    n = int(np.prod(batch))
    m = random_rotations(n).reshape(*batch, 3, 3)
    angles = matrix_to_euler(m, "xyz")
    assert angles.shape == (*batch, 3)
    assert euler_to_matrix(angles, "xyz").shape == (*batch, 3, 3)


def test_gimbal_lock_still_roundtrips_to_the_same_matrix():
    """Pitch at +/-90 degrees is degenerate; the matrix must still survive."""
    angles = torch.tensor(
        [[0.0, torch.pi / 2, 0.0], [0.3, -torch.pi / 2, 0.0]], dtype=torch.float64
    )
    m = euler_to_matrix(angles, "xyz")
    torch.testing.assert_close(
        euler_to_matrix(matrix_to_euler(m, "xyz"), "xyz"), m, atol=1e-8, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype):
    """Both helpers hop through numpy; a hard-coded float64 on the way back
    would silently upcast every camera path."""
    m = random_rotations(4).to(dtype)
    assert matrix_to_euler(m, "xyz").dtype == dtype
    assert euler_to_matrix(matrix_to_euler(m, "xyz"), "xyz").dtype == dtype


# ---------------------------------------------------------------------------
# stabilisation
# ---------------------------------------------------------------------------
def test_a_constant_trajectory_is_returned_unchanged():
    """Gaussian smoothing of a constant signal is that constant, and the
    rebuilt third axis is the original one for a right-handed pose. This is
    the only input whose smoothed value is known in closed form -- and it
    fails immediately if the kernel is unnormalised or the padding is wrong.
    """
    pose = random_se3(1)[0].float()
    poses = pose.expand(11, 4, 4).contiguous()
    out = render_stabilization_path(poses, k_size=5)
    torch.testing.assert_close(out, poses, atol=1e-5, rtol=0)


def test_output_is_homogeneous_and_right_handed():
    poses = random_se3(9).float()
    out = render_stabilization_path(poses, k_size=5)
    assert out.shape == (9, 4, 4)
    torch.testing.assert_close(out[:, 3, :], torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(9, 4))
    r1, r2, r3 = out[:, :3, 0], out[:, :3, 1], out[:, :3, 2]
    torch.testing.assert_close(r1.norm(dim=-1), torch.ones(9), atol=1e-5, rtol=0)
    torch.testing.assert_close(r2.norm(dim=-1), torch.ones(9), atol=1e-5, rtol=0)
    torch.testing.assert_close(r3, torch.linalg.cross(r1, r2), atol=1e-5, rtol=0)


def test_smoothing_actually_smooths_the_camera_centres():
    """A jittered straight line must come out with less frame-to-frame
    movement than it went in with."""
    poses = torch.eye(4).expand(21, 4, 4).contiguous()
    t = torch.linspace(0, 1, 21)
    torch.manual_seed(0)
    poses[:, :3, 3] = torch.stack([t, torch.zeros(21), torch.zeros(21)], dim=-1)
    poses[:, :3, 3] += 0.05 * torch.randn(21, 3)

    out = render_stabilization_path(poses, k_size=9)

    jitter_in = poses[1:, :3, 3].diff(dim=0).norm(dim=-1).sum()
    jitter_out = out[1:, :3, 3].diff(dim=0).norm(dim=-1).sum()
    assert jitter_out < jitter_in / 2


@pytest.mark.parametrize("num_frames", [1, 2, 3])
def test_very_short_trajectories_do_not_crash(num_frames):
    """``k_size`` is clamped to the frame count; reflect padding raises if the
    pad is not smaller than the signal, so the clamping is load-bearing."""
    out = render_stabilization_path(random_se3(num_frames).float(), k_size=45)
    assert out.shape == (num_frames, 4, 4)


def test_3x4_input_is_accepted():
    out = render_stabilization_path(random_se3(7).float()[:, :3, :], k_size=5)
    assert out.shape == (7, 4, 4)


# ---------------------------------------------------------------------------
# render paths
# ---------------------------------------------------------------------------
def test_wander_path_starts_and_ends_at_the_reference_pose():
    c2w = random_se3(1)[0].float()
    k = torch.eye(3)
    c2ws, ks = render_wander_path(c2w, k, h=64, w=64, num_frames=8)
    assert c2ws.shape == (10, 4, 4) and ks.shape == (10, 3, 3)
    torch.testing.assert_close(c2ws[0], c2w)
    torch.testing.assert_close(c2ws[-1], c2w)


def test_dolly_zoom_keeps_the_focus_distance_constant():
    """The point of a dolly zoom: as the camera moves back by z, the focal
    length shrinks by ``D / (D + z)`` so the subject stays the same size."""
    c2w = torch.eye(4)
    k = torch.eye(3)
    k[0, 0] = k[1, 1] = 0.5
    c2ws, ks = render_dolly_zoom_path(
        c2w, k, h=64, w=64, num_frames=16, max_disp=0.3, D_focus=10.0
    )

    z = -c2ws[:, 2, 3]
    expected_fx = (0.5 * 64) * (10.0 / (10.0 + z)) / 64
    torch.testing.assert_close(ks[:, 0, 0], expected_fx, atol=1e-6, rtol=0)


def test_interpolate_intrinsics_is_linear_between_the_endpoints():
    a = torch.eye(3, dtype=torch.float64)
    b = 2 * torch.eye(3, dtype=torch.float64)
    t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    out = interpolate_intrinsics(a, b, t)
    assert out.shape == (3, 3, 3)
    torch.testing.assert_close(out[0], a)
    torch.testing.assert_close(out[1], 1.5 * torch.eye(3, dtype=torch.float64))
    torch.testing.assert_close(out[2], b)


# ---------------------------------------------------------------------------
# ray geometry
# ---------------------------------------------------------------------------
def test_intersect_rays_finds_the_crossing_point():
    """Two rays constructed to meet at (1, 2, 3)."""
    target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    a_origin = torch.zeros(3, dtype=torch.float64)
    b_origin = torch.tensor([4.0, 0.0, 0.0], dtype=torch.float64)
    a_dir = (target - a_origin) / (target - a_origin).norm()
    b_dir = (target - b_origin) / (target - b_origin).norm()
    got = intersect_rays(a_origin, a_dir, b_origin, b_dir)
    torch.testing.assert_close(got, target, atol=1e-9, rtol=0)


def test_generate_coordinate_frame_is_orthonormal_and_right_handed():
    y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    frame = generate_coordinate_frame(y, z)
    torch.testing.assert_close(frame @ frame.mT, torch.eye(3, dtype=torch.float64))
    torch.testing.assert_close(torch.det(frame), torch.tensor(1.0, dtype=torch.float64))
    # The Y column is the supplied y, the Z column the supplied z.
    torch.testing.assert_close(frame[:, 1], y)
    torch.testing.assert_close(frame[:, 2], z)
