"""SO(3) projection and spherical-harmonic rotation in ``utils/sh_helpers.py``.

``project_to_so3_strict`` is checked against the definition it implements --
the Frobenius-nearest rotation, i.e. special orthogonal Procrustes -- not
merely for "the output is a rotation", which any orthonormalisation satisfies
including ones that pick the wrong branch.

``rotate_sh`` needs e3nn, so those tests skip without the ``gs`` extra. They
assert group properties rather than a table of numbers: a Wigner-D
implementation that transposed its rotation, flipped the sign of beta, or
dropped the yzx->xyz basis permutation would still return plausible-looking
coefficients but would stop composing correctly.
"""

import pytest
import torch
from _oracles import nearest_rotation_oracle
from conftest import random_rotations

from depth_anything_3.utils.sh_helpers import project_to_so3_strict


def _perturbed(n=32, sigma=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    return random_rotations(n, seed=seed) + sigma * torch.randn(
        n, 3, 3, generator=g, dtype=torch.float64
    )


# ---------------------------------------------------------------------------
# project_to_so3_strict
# ---------------------------------------------------------------------------
def test_projection_is_the_identity_on_rotations():
    r = random_rotations(32)
    torch.testing.assert_close(project_to_so3_strict(r), r)


def test_projection_matches_special_orthogonal_procrustes():
    """The independent judge: ``U diag(1, 1, det(UV^T)) V^T``."""
    m = _perturbed()
    torch.testing.assert_close(
        project_to_so3_strict(m), torch.from_numpy(nearest_rotation_oracle(m.numpy()))
    )


def test_projection_is_optimal_not_merely_orthonormal():
    """``argmax_R trace(R^T M)`` over SO(3). Checked by sampling: no random
    rotation may score better than the returned one."""
    m = _perturbed(n=8, seed=1)
    best = project_to_so3_strict(m)
    best_score = torch.einsum("bij,bij->b", best, m)
    candidates = random_rotations(256, seed=99)
    for candidate in candidates:
        score = torch.einsum("ij,bij->b", candidate, m)
        assert torch.all(score <= best_score + 1e-9)


def test_projection_output_is_in_so3():
    out = project_to_so3_strict(_perturbed())
    torch.testing.assert_close(out @ out.mT, torch.eye(3, dtype=torch.float64).expand(32, 3, 3))
    torch.testing.assert_close(torch.det(out), torch.ones(32, dtype=torch.float64))


def test_projection_repairs_a_reflection():
    """A matrix with det = -1 must come back with det = +1.

    Note an exact reflection has three equal singular values, so the nearest
    rotation is genuinely non-unique -- only the determinant can be asserted
    here. The next test perturbs it to break the tie and then pins the value.
    """
    r = random_rotations(8).clone()
    r[:, :, 0] *= -1  # now a reflection
    assert torch.all(torch.det(r) < 0)
    out = project_to_so3_strict(r)
    torch.testing.assert_close(torch.det(out), torch.ones(8, dtype=torch.float64))


def test_projection_of_a_non_degenerate_reflection_picks_the_nearest_rotation():
    """Which axis to flip is the decision this function exists to make. With
    distinct singular values there is exactly one right answer."""
    g = torch.Generator().manual_seed(4)
    r = random_rotations(8, seed=4).clone()
    r[:, :, 0] *= -1
    m = r + 0.2 * torch.randn(8, 3, 3, generator=g, dtype=torch.float64)
    assert torch.all(torch.det(m) < 0)
    torch.testing.assert_close(
        project_to_so3_strict(m), torch.from_numpy(nearest_rotation_oracle(m.numpy()))
    )


@pytest.mark.parametrize("batch", [(), (4,), (2, 3)])
def test_projection_preserves_batch_shape(batch):
    m = torch.eye(3, dtype=torch.float64).expand(*batch, 3, 3)
    assert project_to_so3_strict(m).shape == (*batch, 3, 3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_projection_preserves_dtype(dtype):
    """It builds correction tensors explicitly; a hard-coded float32 there
    would silently downcast the float64 callers in ``rotate_sh``."""
    assert project_to_so3_strict(random_rotations(4).to(dtype)).dtype == dtype


def test_projection_rejects_wrong_shape():
    with pytest.raises(ValueError):
        project_to_so3_strict(torch.zeros(4, 2, 2))


# ---------------------------------------------------------------------------
# rotate_sh
# ---------------------------------------------------------------------------
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


def test_the_error_message_names_the_extra_that_provides_it(monkeypatch):
    from depth_anything_3.utils import sh_helpers

    monkeypatch.setattr(sh_helpers, "E3NN_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"\[gs\]"):
        sh_helpers.rotate_sh(torch.zeros(1, 4), torch.eye(3).unsqueeze(0))


@pytest.fixture
def rotate_sh():
    pytest.importorskip("e3nn", reason="rotate_sh needs the `gs` extra")
    from depth_anything_3.utils.sh_helpers import rotate_sh as fn

    return fn


def _sh(n=9, batch=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n, generator=g)


def test_rotating_by_the_identity_changes_nothing(rotate_sh):
    coefficients = _sh()
    eye = torch.eye(3).expand(5, 3, 3)
    torch.testing.assert_close(rotate_sh(coefficients, eye), coefficients, atol=1e-5, rtol=0)


def test_rotate_sh_composes_like_the_rotations_do(rotate_sh):
    """``rotate_sh(rotate_sh(c, B), A) == rotate_sh(c, A @ B)``.

    A transposed Wigner-D or a sign error on beta gives an *anti*-homomorphism
    -- correct for a single rotation, wrong the moment two compose. That is
    the failure mode an e3nn version bump would introduce."""
    coefficients = _sh()
    a = random_rotations(5, seed=1).float()
    b = random_rotations(5, seed=2).float()
    torch.testing.assert_close(
        rotate_sh(rotate_sh(coefficients, b), a),
        rotate_sh(coefficients, a @ b),
        atol=1e-4,
        rtol=1e-4,
    )


def test_the_dc_band_is_rotation_invariant(rotate_sh):
    """Degree 0 is a constant function on the sphere; no rotation touches it.
    In ``gs_adapter`` that band carries the gaussian's base colour."""
    coefficients = _sh()
    rotated = rotate_sh(coefficients, random_rotations(5, seed=3).float())
    torch.testing.assert_close(rotated[:, :1], coefficients[:, :1], atol=1e-6, rtol=0)


@pytest.mark.parametrize("degree", [1, 2])
def test_each_degree_band_keeps_its_norm(rotate_sh, degree):
    """Wigner-D matrices are orthogonal, so rotation redistributes energy
    within a band but never between bands."""
    coefficients = _sh()
    rotated = rotate_sh(coefficients, random_rotations(5, seed=4).float())
    lo, hi = degree**2, (degree + 1) ** 2
    torch.testing.assert_close(
        rotated[:, lo:hi].norm(dim=-1),
        coefficients[:, lo:hi].norm(dim=-1),
        atol=1e-4,
        rtol=1e-4,
    )


def test_rotate_sh_is_linear_in_the_coefficients(rotate_sh):
    a, b = _sh(seed=5), _sh(seed=6)
    r = random_rotations(5, seed=7).float()
    torch.testing.assert_close(
        rotate_sh(2 * a + 3 * b, r),
        2 * rotate_sh(a, r) + 3 * rotate_sh(b, r),
        atol=1e-4,
        rtol=1e-4,
    )
