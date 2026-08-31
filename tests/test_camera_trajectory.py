"""Euler-angle conversions in utils/camera_trj_helpers.py.

Both helpers delegate to scipy and move between torch and numpy, so the
risk is in the dtype/device/shape plumbing rather than the maths.
"""

import pytest
import torch
from conftest import random_rotations

from depth_anything_3.utils.camera_trj_helpers import euler_to_matrix, matrix_to_euler

PATTERNS = ["xyz", "zyx", "ZYX"]


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
    n = 1
    for b in batch:
        n *= b
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


def test_dtype_is_preserved():
    m = random_rotations(4).to(torch.float32)
    assert matrix_to_euler(m, "xyz").dtype == torch.float32
