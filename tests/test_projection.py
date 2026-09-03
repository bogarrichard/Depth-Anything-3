"""Camera projection contracts in ``utils/geometry.py``.

The pinhole helpers here are the arithmetic behind every exported point
cloud, so the tests check them against a hand-written projection oracle
rather than against each other. Two conventions are asserted explicitly
because they are inconsistent *within this module* and a well-meaning
refactor would unify them and shift every reconstruction:

  * ``sample_image_grid`` yields **normalised pixel centres**, ``(i + 0.5) / n``;
  * ``unproject_depth`` builds its grid from **integer pixel indices**, no
    half-pixel offset.

``get_world_rays`` takes a **camera-to-world** matrix despite its parameter
being called ``extrinsics``; that is asserted too.
"""

import numpy as np
import pytest
import torch
from _oracles import pinhole_matrix, project_oracle, unproject_oracle
from conftest import random_se3

from depth_anything_3.utils.geometry import (
    camera_space_to_pixel_space,
    camera_space_to_world_space,
    get_fov,
    get_world_rays,
    homogenize_points,
    homogenize_vectors,
    map_pdf_to_opacity,
    normalize_homogenous_points,
    pixel_space_to_camera_space,
    sample_image_grid,
    transform_cam2world,
    unproject,
    unproject_depth,
    world_space_to_camera_space,
)

H, W = 6, 8


def _intrinsics(fx=5.0, fy=4.0, cx=W / 2, cy=H / 2) -> torch.Tensor:
    return torch.from_numpy(pinhole_matrix(fx, fy, cx, cy))


# ---------------------------------------------------------------------------
# grids and homogeneous coordinates
# ---------------------------------------------------------------------------
def test_sample_image_grid_returns_normalised_pixel_centres():
    coords, indices = sample_image_grid((H, W))
    assert coords.shape == (H, W, 2) and indices.shape == (H, W, 2)
    # coords[..., 0] runs along the width, coords[..., 1] along the height.
    expected_x = (torch.arange(W) + 0.5) / W
    expected_y = (torch.arange(H) + 0.5) / H
    torch.testing.assert_close(coords[0, :, 0], expected_x)
    torch.testing.assert_close(coords[:, 0, 1], expected_y)
    # The integer companion is (row, col), i.e. ij indexing.
    torch.testing.assert_close(indices[:, 0, 0], torch.arange(H))
    torch.testing.assert_close(indices[0, :, 1], torch.arange(W))


def test_homogenize_points_and_vectors_differ_only_in_the_last_entry():
    p = torch.randn(4, 3, dtype=torch.float64)
    torch.testing.assert_close(homogenize_points(p)[..., :3], p)
    torch.testing.assert_close(homogenize_points(p)[..., 3], torch.ones(4, dtype=torch.float64))
    torch.testing.assert_close(homogenize_vectors(p)[..., 3], torch.zeros(4, dtype=torch.float64))


def test_homogenize_distinguishes_points_from_vectors_under_translation():
    """A translation must move a point and leave a direction alone."""
    transform = random_se3(1)[0]
    p = torch.randn(4, 3, dtype=torch.float64)
    moved_point = transform_cam2world(homogenize_points(p), transform)
    moved_vector = transform_cam2world(homogenize_vectors(p), transform)
    torch.testing.assert_close(
        moved_point[..., :3] - moved_vector[..., :3], transform[:3, 3].expand(4, 3)
    )


def test_normalize_homogenous_points_divides_by_the_last_coordinate():
    x = torch.tensor([[2.0, 4.0, 2.0], [3.0, -6.0, 3.0]], dtype=torch.float64)
    torch.testing.assert_close(
        normalize_homogenous_points(x),
        torch.tensor([[1.0, 2.0, 1.0], [1.0, -2.0, 1.0]], dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# unproject / rays
# ---------------------------------------------------------------------------
def test_unproject_matches_the_pinhole_oracle():
    k = _intrinsics()
    coords = torch.tensor([[0.0, 0.0], [3.0, 2.0], [7.0, 5.0]], dtype=torch.float64)
    z = torch.tensor([1.0, 2.5, 0.75], dtype=torch.float64)
    expected = torch.from_numpy(unproject_oracle(coords.numpy(), z.numpy(), k.numpy()))
    # `unproject` inverts the intrinsics in float32 internally, hence the tolerance.
    torch.testing.assert_close(unproject(coords, z, k), expected, atol=1e-5, rtol=0)


def test_get_world_rays_takes_camera_to_world_not_world_to_camera():
    """The parameter is named ``extrinsics`` but is applied directly, so it
    must be the camera-to-world matrix. ``gs_adapter`` passes ``cam2worlds``;
    reading it as world-to-camera would mirror every scene."""
    k = _intrinsics()
    c2w = random_se3(1)[0]
    coords = torch.tensor([[3.0, 2.0]], dtype=torch.float64)

    origins, directions = get_world_rays(coords, c2w, k)

    # Origin is the camera centre in world space, i.e. the c2w translation.
    torch.testing.assert_close(origins, c2w[:3, 3].expand(1, 3))
    # Direction is the camera-space ray rotated into world space.
    ray_cam = torch.from_numpy(unproject_oracle(coords.numpy(), np.ones(1), k.numpy()))
    ray_cam = ray_cam / ray_cam.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(directions, ray_cam @ c2w[:3, :3].mT, atol=1e-5, rtol=0)


def test_get_world_rays_returns_unit_directions():
    k = _intrinsics()
    c2w = random_se3(1)[0]
    coords, _ = sample_image_grid((H, W))
    _, directions = get_world_rays(coords.to(torch.float64), c2w, k)
    torch.testing.assert_close(
        directions.norm(dim=-1), torch.ones(H, W, dtype=torch.float64), atol=1e-5, rtol=0
    )


def test_get_fov_matches_the_closed_form_for_normalised_intrinsics():
    """``get_fov`` probes the image edges at 0 and 1, so it expects intrinsics
    normalised to a unit-square image."""
    fx, fy = 0.8, 1.3
    k = _intrinsics(fx=fx, fy=fy, cx=0.5, cy=0.5)[None]
    fov = get_fov(k)
    torch.testing.assert_close(
        fov,
        torch.tensor([[2 * np.arctan(0.5 / fx), 2 * np.arctan(0.5 / fy)]], dtype=torch.float64),
        atol=1e-6,
        rtol=0,
    )


# ---------------------------------------------------------------------------
# the pixel -> camera -> world chain
# ---------------------------------------------------------------------------
def test_pixel_to_camera_space_matches_the_oracle():
    k = _intrinsics()[None, None]  # (b, v, 3, 3)
    pixels, _ = sample_image_grid((H, W))
    pixels = pixels.to(torch.float64)
    depth = torch.rand(1, 1, H, W, 1, dtype=torch.float64) + 0.5

    out = pixel_space_to_camera_space(pixels, depth, k)

    expected = unproject_oracle(pixels.numpy(), depth[0, 0, ..., 0].numpy(), k[0, 0].numpy())
    torch.testing.assert_close(out[0, 0], torch.from_numpy(expected), atol=1e-9, rtol=0)


def test_camera_and_world_space_are_inverses():
    """world_space_to_camera_space must undo camera_space_to_world_space for
    the matching view. It broadcasts to (b, v1, v2, ...), so the round trip
    lives on the diagonal."""
    c2w = random_se3(2)[None]  # (1, 2, 4, 4)
    cam = torch.randn(1, 2, H, W, 3, dtype=torch.float64)

    world = camera_space_to_world_space(cam, c2w)
    back = world_space_to_camera_space(world, c2w)

    for v in range(2):
        torch.testing.assert_close(back[0, v, v], cam[0, v])


def test_full_pixel_roundtrip_recovers_the_pixel_grid():
    k = _intrinsics().expand(1, 2, 3, 3).contiguous()
    c2w = random_se3(2)[None]
    pixels, _ = sample_image_grid((H, W))
    pixels = pixels.to(torch.float64) * torch.tensor([W, H], dtype=torch.float64)
    depth = torch.rand(1, 2, H, W, 1, dtype=torch.float64) + 0.5

    cam = pixel_space_to_camera_space(pixels, depth, k)
    world = camera_space_to_world_space(cam, c2w)
    back_cam = world_space_to_camera_space(world, c2w)
    back_pixels = camera_space_to_pixel_space(back_cam, k)

    for v in range(2):
        torch.testing.assert_close(back_pixels[0, v, v], pixels, atol=1e-8, rtol=0)


def test_camera_space_to_pixel_space_matches_the_oracle():
    k = _intrinsics()[None, None]
    cam = torch.rand(1, 1, 1, 2, 3, 3, dtype=torch.float64) + torch.tensor(
        [0.0, 0.0, 2.0], dtype=torch.float64
    )
    out = camera_space_to_pixel_space(cam, k)
    expected, _ = project_oracle(cam[0, 0, 0].numpy(), k[0, 0].numpy())
    torch.testing.assert_close(out[0, 0, 0], torch.from_numpy(expected), atol=1e-9, rtol=0)


# ---------------------------------------------------------------------------
# unproject_depth
# ---------------------------------------------------------------------------
def test_unproject_depth_uses_integer_pixel_indices():
    """No half-pixel offset here, unlike ``sample_image_grid``. Adding one
    would move every exported point by half a pixel of parallax."""
    k = _intrinsics()[None, None]
    depth = torch.rand(1, 1, H, W, 1, dtype=torch.float64) + 0.5

    out = unproject_depth(depth, k)  # c2w defaults to the identity

    grid = np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), axis=-1)
    expected = unproject_oracle(
        grid.astype(np.float64), depth[0, 0, ..., 0].numpy(), k[0, 0].numpy()
    )
    torch.testing.assert_close(out[0, 0], torch.from_numpy(expected), atol=1e-9, rtol=0)


def test_unproject_depth_applies_the_camera_to_world_transform():
    k = _intrinsics()[None, None]
    c2w = random_se3(1)[None]
    depth = torch.rand(1, 1, H, W, 1, dtype=torch.float64) + 0.5

    world = unproject_depth(depth, k, c2w)
    cam = unproject_depth(depth, k)

    expected = camera_space_to_world_space(cam, c2w)
    torch.testing.assert_close(world, expected)


def test_unproject_depth_recovers_depth_when_projected_back():
    k = _intrinsics()[None, None]
    c2w = random_se3(1)[None]
    depth = torch.rand(1, 1, H, W, 1, dtype=torch.float64) + 0.5

    world = unproject_depth(depth, k, c2w)
    back = world_space_to_camera_space(world, c2w)[0, 0, 0]

    torch.testing.assert_close(back[..., 2], depth[0, 0, ..., 0], atol=1e-9, rtol=0)


# ---------------------------------------------------------------------------
# opacity mapping
# ---------------------------------------------------------------------------
def test_map_pdf_to_opacity_is_the_identity_without_a_mapping():
    """With no schedule the exponent is 1 and the formula collapses to ``pdf``."""
    pdf = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    torch.testing.assert_close(map_pdf_to_opacity(pdf), pdf)


@pytest.mark.parametrize("step", [0, 5, 100])
def test_map_pdf_to_opacity_fixes_the_endpoints_and_stays_monotone(step):
    mapping = {"initial": -1.0, "final": 2.0, "warm_up": 10}
    pdf = torch.linspace(0.0, 1.0, 65, dtype=torch.float64)
    out = map_pdf_to_opacity(pdf, global_step=step, opacity_mapping=mapping)
    torch.testing.assert_close(out[0], torch.zeros((), dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(out[-1], torch.ones((), dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.all(out.diff() > 0)
