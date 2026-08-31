"""Shared fixtures and helpers for the test suite.

Everything here is CPU-only and needs no model weights: the target is the
pure-function geometry, packaging and CLI surface.
"""

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    """Make every test reproducible regardless of execution order."""
    torch.manual_seed(0)
    np.random.seed(0)


def random_rotations(n: int, *, seed: int = 0) -> torch.Tensor:
    """`n` uniformly distributed rotation matrices, shape (n, 3, 3).

    Built by QR-decomposing gaussian matrices and fixing the sign so the
    determinant is +1, which keeps the result in SO(3) rather than O(3).
    """
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, 3, 3, generator=g, dtype=torch.float64))
    # QR is only unique up to the signs of R's diagonal; normalise them.
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    # Flip one column of any reflection so det becomes +1.
    flip = torch.det(q) < 0
    q[flip, :, 0] = -q[flip, :, 0]
    return q


def random_se3(n: int, *, seed: int = 0) -> torch.Tensor:
    """`n` homogeneous 4x4 rigid transforms, shape (n, 4, 4)."""
    g = torch.Generator().manual_seed(seed + 1)
    out = torch.eye(4, dtype=torch.float64).repeat(n, 1, 1)
    out[:, :3, :3] = random_rotations(n, seed=seed)
    out[:, :3, 3] = torch.randn(n, 3, generator=g, dtype=torch.float64)
    return out
