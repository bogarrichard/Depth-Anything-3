"""`utils/geometry.py` and `model/utils/transform.py` carry near-identical
copies of the quaternion helpers.

These tests pin the two implementations to each other. They are the safety
net for merging the modules: if the copies have already drifted, this fails
now rather than after the refactor, and if they agree, the refactor can be
made with the tests staying green.
"""

import torch
from conftest import random_rotations

from depth_anything_3.model.utils import transform as t
from depth_anything_3.utils import geometry as g


def test_quat_to_mat_copies_agree():
    q = torch.nn.functional.normalize(torch.randn(64, 4, dtype=torch.float64), dim=-1)
    torch.testing.assert_close(g.quat_to_mat(q), t.quat_to_mat(q))


def test_mat_to_quat_copies_agree():
    m = random_rotations(64)
    torch.testing.assert_close(g.mat_to_quat(m), t.mat_to_quat(m))


def test_standardize_quaternion_copies_agree():
    q = torch.nn.functional.normalize(torch.randn(64, 4, dtype=torch.float64), dim=-1)
    torch.testing.assert_close(g.standardize_quaternion(q), t.standardize_quaternion(q))
