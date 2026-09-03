"""SO(3) projection in utils/sh_helpers.py.

The module imports e3nn lazily behind try/except, so this is importable
without the `gs` extra.
"""

import pytest
import torch
from conftest import random_rotations

from depth_anything_3.utils.sh_helpers import project_to_so3_strict


def test_projection_is_identity_on_rotations():
    r = random_rotations(32)
    torch.testing.assert_close(project_to_so3_strict(r), r)


def test_projection_of_perturbed_rotation_is_a_rotation():
    r = random_rotations(32)
    noisy = r + 0.05 * torch.randn(32, 3, 3, dtype=torch.float64)
    out = project_to_so3_strict(noisy)
    torch.testing.assert_close(out @ out.mT, torch.eye(3, dtype=torch.float64).expand(32, 3, 3))
    torch.testing.assert_close(torch.det(out), torch.ones(32, dtype=torch.float64))


def test_projection_repairs_a_reflection():
    """A matrix with det = -1 must come back with det = +1."""
    r = random_rotations(8).clone()
    r[:, :, 0] *= -1  # now a reflection
    assert torch.all(torch.det(r) < 0)
    out = project_to_so3_strict(r)
    torch.testing.assert_close(torch.det(out), torch.ones(8, dtype=torch.float64))


def test_projection_rejects_wrong_shape():
    with pytest.raises(ValueError):
        project_to_so3_strict(torch.zeros(4, 2, 2))


def test_rotate_sh_raises_a_clear_error_without_e3nn(monkeypatch):
    """Without e3nn the old code hit a bare NameError deep inside rotate_sh.

    The names were only bound inside the successful branch of the import
    guard, while rotate_sh called them unconditionally. This is reachable in
    practice: gs_adapter calls rotate_sh whenever sh_degree > 0, and
    configs/da3-giant.yaml sets sh_degree: 2.
    """
    from depth_anything_3.utils import sh_helpers

    monkeypatch.setattr(sh_helpers, "E3NN_AVAILABLE", False)
    with pytest.raises(ImportError, match="e3nn"):
        sh_helpers.rotate_sh(torch.zeros(1, 4), torch.eye(3).unsqueeze(0))
