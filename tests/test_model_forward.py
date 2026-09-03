"""The real forward pass of ``DepthAnything3Net``.

Nothing else in the suite runs the network, and a randomly initialised
``da3-small`` builds in well under a second on CPU -- so there is no excuse
for leaving the thing the package exists for untested. These tests do not
assert *values* (the weights are random); they assert the **output contract**:
which keys exist, what shape and dtype they have, and the mathematical
guarantees that the head activations and the pose decoder are supposed to
provide. That is exactly what a torch upgrade, an einops change or a head
refactor breaks.
"""

import pytest
import torch
from conftest import TINY_HW

H, W = TINY_HW
PATCH = 14


def _images(views: int, batch: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.randn(batch, views, 3, H, W, generator=g)


@pytest.fixture(scope="module")
def two_view_output(tiny_net):
    with torch.no_grad():
        return tiny_net(_images(2))


# ---------------------------------------------------------------------------
# output contract
# ---------------------------------------------------------------------------
def test_forward_returns_the_documented_keys(two_view_output):
    assert set(two_view_output.keys()) == {
        "depth",
        "depth_conf",
        "extrinsics",
        "intrinsics",
        "aux",
    }


def test_forward_output_shapes(two_view_output):
    assert two_view_output.depth.shape == (1, 2, H, W)
    assert two_view_output.depth_conf.shape == (1, 2, H, W)
    assert two_view_output.extrinsics.shape == (1, 2, 3, 4)
    assert two_view_output.intrinsics.shape == (1, 2, 3, 3)


def test_forward_output_is_float32_and_finite(two_view_output):
    for key in ("depth", "depth_conf", "extrinsics", "intrinsics"):
        assert two_view_output[key].dtype == torch.float32, key
        assert torch.isfinite(two_view_output[key]).all(), key


def test_depth_is_strictly_positive(two_view_output):
    """The head's ``activation="exp"`` guarantees it. Swapping to a linear or
    tanh activation would let negative depths through and quietly break every
    unprojection downstream."""
    assert torch.all(two_view_output.depth > 0)


def test_confidence_is_greater_than_one(two_view_output):
    """``conf_activation="expp1"`` is ``exp(x) + 1``; the GLB exporter's
    default confidence threshold of 1.05 is calibrated to that floor."""
    assert torch.all(two_view_output.depth_conf > 1.0)


def test_predicted_extrinsics_have_a_proper_rotation_block(two_view_output):
    r = two_view_output.extrinsics[..., :3, :3].double()
    eye = torch.eye(3, dtype=torch.float64).expand_as(r)
    torch.testing.assert_close(r @ r.mT, eye, atol=1e-5, rtol=0)
    torch.testing.assert_close(
        torch.det(r), torch.ones(1, 2, dtype=torch.float64), atol=1e-5, rtol=0
    )


def test_predicted_intrinsics_are_a_centred_pinhole(two_view_output):
    k = two_view_output.intrinsics
    assert torch.all(k[..., 0, 0] > 0) and torch.all(k[..., 1, 1] > 0)
    assert torch.all(k[..., 0, 2] == W / 2) and torch.all(k[..., 1, 2] == H / 2)
    assert torch.all(k[..., 2, 2] == 1.0)
    for i, j in [(0, 1), (1, 0), (2, 0), (2, 1)]:
        assert torch.all(k[..., i, j] == 0.0), (i, j)


@pytest.mark.parametrize("views", [1, 2, 3, 5])
def test_forward_accepts_any_number_of_views(tiny_net, views):
    """S == 3 is the threshold where reference-view reordering switches on
    (``THRESH_FOR_REF_SELECTION``), so both sides of it are exercised."""
    with torch.no_grad():
        out = tiny_net(_images(views))
    assert out.depth.shape == (1, views, H, W)
    assert torch.isfinite(out.depth).all()


# ---------------------------------------------------------------------------
# determinism and batch independence
# ---------------------------------------------------------------------------
def test_forward_is_deterministic(tiny_net):
    x = _images(3)
    with torch.no_grad():
        a, b = tiny_net(x), tiny_net(x)
    for key in ("depth", "depth_conf", "extrinsics", "intrinsics"):
        assert torch.equal(a[key], b[key]), key


def test_batch_items_do_not_leak_into_each_other(tiny_net):
    """Attention is global across *views*, not across the batch. If a
    rearrange ever collapses B into S, this is what catches it."""
    x0, x1 = _images(2), _images(2) * 0.5
    with torch.no_grad():
        separate = [tiny_net(x0), tiny_net(x1)]
        together = tiny_net(torch.cat([x0, x1], dim=0))
    for i, single in enumerate(separate):
        torch.testing.assert_close(together.depth[i : i + 1], single.depth, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# reference view selection
# ---------------------------------------------------------------------------
def test_unknown_reference_view_strategy_is_rejected(tiny_net):
    """``--ref-view-strategy`` is a free string on the CLI, so a typo must
    fail loudly rather than fall back to some default."""
    with torch.no_grad(), pytest.raises(ValueError, match="strategy"):
        tiny_net(_images(3), ref_view_strategy="not-a-strategy")


def test_reference_view_selection_is_skipped_below_the_threshold(tiny_net):
    """Documented: with two or fewer views no reordering happens at all --
    which is why an invalid strategy is not even looked at there."""
    with torch.no_grad():
        out = tiny_net(_images(2), ref_view_strategy="not-a-strategy")
    assert torch.isfinite(out.depth).all()


@pytest.mark.parametrize("strategy", ["first", "middle", "saddle_balanced", "saddle_sim_range"])
def test_every_advertised_strategy_runs(tiny_net, strategy):
    with torch.no_grad():
        out = tiny_net(_images(4), ref_view_strategy=strategy)
    assert torch.isfinite(out.depth).all()


def test_reordering_is_undone_before_the_output(tiny_net):
    """The backbone moves the chosen reference view to slot 0 and is supposed
    to put everything back before the heads run.

    The check is exact, not approximate: ``middle`` on the original views has
    to compute the same thing as ``first`` on views already in that order,
    with the output permuted back. If ``restore_original_order`` stopped
    being the inverse of ``reorder_by_reference``, every depth map would be
    attributed to the wrong input image -- a failure no shape or finiteness
    assertion notices.
    """
    x = _images(4)
    perm = [2, 0, 1, 3]  # what "middle" selects internally for S == 4
    inverse = [perm.index(i) for i in range(4)]
    with torch.no_grad():
        reordered_internally = tiny_net(x, ref_view_strategy="middle")
        reordered_by_hand = tiny_net(x[:, perm], ref_view_strategy="first")
    assert torch.equal(reordered_internally.depth, reordered_by_hand.depth[:, inverse])
    torch.testing.assert_close(
        reordered_internally.extrinsics,
        reordered_by_hand.extrinsics[:, inverse],
        atol=1e-6,
        rtol=0,
    )


# ---------------------------------------------------------------------------
# optional branches
# ---------------------------------------------------------------------------
def test_supplying_camera_poses_takes_the_camera_encoder_path(tiny_net):
    """With extrinsics given, ``cam_enc`` produces the camera tokens instead
    of the learned ones -- and the reference-view reordering is skipped."""
    from conftest import random_se3

    ext = random_se3(3).float()[None]
    k = torch.eye(3).expand(1, 3, 3, 3).contiguous()
    with torch.no_grad():
        out = tiny_net(_images(3), extrinsics=ext, intrinsics=k)
    assert out.depth.shape == (1, 3, H, W)
    assert torch.isfinite(out.depth).all()


def test_export_feat_layers_returns_patch_grids(tiny_net):
    layers = [3, 7]
    with torch.no_grad():
        out = tiny_net(_images(2), export_feat_layers=layers)
    assert set(out.aux.keys()) == {f"feat_layer_{i}" for i in layers}
    for value in out.aux.values():
        assert value.shape[:4] == (1, 2, H // PATCH, W // PATCH)
        assert torch.isfinite(value).all()


def test_no_feature_layers_requested_means_no_aux(two_view_output):
    assert dict(two_view_output.aux) == {}


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
def test_non_patch_multiple_input_fails_loudly(tiny_net):
    """504/14 is exact for every shipped preset; a size that is not a
    multiple of the patch size must raise rather than silently crop."""
    with torch.no_grad(), pytest.raises(Exception):
        tiny_net(torch.randn(1, 2, 3, H + 1, W))


# ---------------------------------------------------------------------------
# the head's chunking optimisation
# ---------------------------------------------------------------------------
def test_dpt_head_chunking_does_not_change_the_result(tiny_net):
    """``DualDPT.forward`` splits the batch when ``S > chunk_size``. The
    chunked and unchunked paths are separate code; they must agree."""
    x = _images(4)
    with torch.no_grad():
        feats, _ = tiny_net.backbone(x, cam_token=None, export_feat_layers=[])
        whole = tiny_net.head(feats, H, W, patch_start_idx=0, chunk_size=None)
        chunked = tiny_net.head(feats, H, W, patch_start_idx=0, chunk_size=2)
    assert set(whole.keys()) == set(chunked.keys())
    for key in whole:
        torch.testing.assert_close(whole[key], chunked[key], atol=1e-6, rtol=1e-6)


def test_camera_decoder_consumes_the_camera_token(tiny_net):
    """``feats[-1][1]`` is the per-view camera token, not the patch grid.
    Getting that index wrong is a silent shape-compatible mistake only if the
    token width happens to match, so it is pinned here."""
    x = _images(2)
    with torch.no_grad():
        feats, _ = tiny_net.backbone(x, cam_token=None, export_feat_layers=[])
        pose_enc = tiny_net.cam_dec(feats[-1][1])
    assert feats[-1][1].shape[:2] == (1, 2)
    assert pose_enc.shape == (1, 2, 9)
    assert torch.isfinite(pose_enc).all()
    # The last two channels are field-of-view angles behind a ReLU.
    assert torch.all(pose_enc[..., -2:] >= 0)


def test_predicted_intrinsics_follow_from_the_pose_encoding(tiny_net):
    """End-to-end check that ``_process_camera_estimation`` really is
    ``pose_encoding_to_extri_intri`` followed by ``affine_inverse``."""
    from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
    from depth_anything_3.utils.geometry import affine_inverse

    x = _images(2)
    with torch.no_grad():
        out = tiny_net(x)
        feats, _ = tiny_net.backbone(x, cam_token=None, export_feat_layers=[])
        c2w, k = pose_encoding_to_extri_intri(tiny_net.cam_dec(feats[-1][1]), (H, W))
    torch.testing.assert_close(out.intrinsics, k)
    # The head decodes a camera-to-world pose; the model publishes its inverse.
    torch.testing.assert_close(out.extrinsics, affine_inverse(c2w))
    assert not torch.allclose(out.extrinsics, c2w)
