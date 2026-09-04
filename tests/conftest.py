"""Shared fixtures and helpers for the test suite.

What this suite is for
----------------------
These tests are meant to be a *critic* of the implementation, not a mirror of
it. Concretely, three rules shape every file here:

1. **Judge against something independent.** Expected values come from closed
   form maths, from scipy, or from the reference implementations in
   ``tests/_oracles.py`` -- never from running the code under test and
   recording what it printed. A round trip alone (``f(g(x)) == x``) is only
   ever a supporting check, because a self-consistently wrong pair passes it.

2. **Pin the contract, not the code path.** Conventions (scalar-last
   quaternions, world-to-camera extrinsics, the ``__object__`` config
   protocol, the export format table, which keyword reaches which function)
   are asserted explicitly so a library swap, a version bump or a refactor
   that changes behaviour fails loudly instead of silently producing
   different numbers.

3. **Exercise the real mechanism.** The model, the config system, the
   input/output processors and the export dispatcher are run for real on a
   small randomly-initialised network. No weights, no GPU, no network access.

Everything here is CPU-only and needs no model weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# The smallest shipped preset. Its backbone is 34M parameters, which builds in
# well under a second on CPU, so the real network can be exercised in tests.
TINY_PRESET = "da3-small"
# Both must be multiples of DepthAnything3Net.PATCH_SIZE (14).
TINY_HW = (56, 70)


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


def random_unit_quaternions(n: int, *, seed: int = 0) -> torch.Tensor:
    """`n` unit scalar-last (XYZW) quaternions, shape (n, 4)."""
    g = torch.Generator().manual_seed(seed + 2)
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


@pytest.fixture(scope="session")
def tiny_net():
    """A real ``DepthAnything3Net`` built from the smallest shipped preset.

    Randomly initialised: the point is to exercise the wiring, shapes and
    dtypes of the forward pass, which no amount of pure-function testing
    covers and which is exactly what a torch upgrade breaks.
    """
    from depth_anything_3.cfg import create_object, load_config

    torch.manual_seed(0)
    net = create_object(load_config(f"depth_anything_3.configs.{TINY_PRESET}"))
    net.eval()
    return net


@pytest.fixture(scope="session")
def tiny_api_model():
    """The public ``DepthAnything3`` wrapper around the smallest preset."""
    from depth_anything_3.api import DepthAnything3

    torch.manual_seed(0)
    return DepthAnything3(model_name=TINY_PRESET)


@pytest.fixture(scope="session")
def tiny_checkpoint_dir(tmp_path_factory, tiny_api_model):
    """``tiny_api_model`` saved to a local directory via ``save_pretrained``.

    The CLI always loads through ``DepthAnything3.from_pretrained(model_dir)``
    (never the ``model_name=`` constructor), so CLI-level tests need a real
    on-disk checkpoint -- this is the local-directory form of that, taking
    the same round trip ``huggingface_hub.PyTorchModelHubMixin`` uses to load
    from the Hub, minus the network call.
    """
    directory = tmp_path_factory.mktemp("tiny_checkpoint")
    tiny_api_model.save_pretrained(str(directory))
    return str(directory)


@pytest.fixture(scope="session")
def image_files(tmp_path_factory):
    """Three small on-disk RGB images with distinguishable content."""
    from PIL import Image

    directory = tmp_path_factory.mktemp("images")
    rng = np.random.default_rng(0)
    paths = []
    for i in range(3):
        # Smooth-ish content rather than pure noise, so resizing is meaningful.
        base = rng.integers(0, 255, size=(4, 5, 3), dtype=np.uint8)
        arr = np.kron(base, np.ones((16, 16, 1), dtype=np.uint8))
        path = directory / f"view_{i}.png"
        Image.fromarray(arr).save(path)
        paths.append(str(path))
    return paths
