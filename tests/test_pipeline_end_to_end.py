"""``DepthAnything3.inference`` from image files to exported artefacts.

Everything below is run for real on a randomly initialised ``da3-small``:
files are read from disk, resized, normalised, pushed through the network,
converted to a ``Prediction`` and written out. The numbers are meaningless --
the weights are random -- but the *pipeline* is not, and this is the only
test in the suite that proves the pieces still fit together.

The individual stages that carry geometry (extrinsic normalisation, the
alignment onto user-supplied poses, the image denormalisation) are checked
separately against their definitions, because an end-to-end smoke test alone
would happily pass with any of them silently wrong.
"""

import time
from pathlib import Path
import numpy as np
import pytest
import torch

from depth_anything_3.specs import Prediction

PROCESS_RES = 56


@pytest.fixture(scope="module")
def prediction(tiny_api_model, image_files):
    return tiny_api_model.inference(
        image_files, process_res=PROCESS_RES, process_res_method="upper_bound_resize"
    )


def _wait_for(path: Path, timeout: float = 60.0) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:
            time.sleep(0.05)
            return path
        time.sleep(0.02)
    raise AssertionError(f"{path} was never written")


# ---------------------------------------------------------------------------
# the prediction
# ---------------------------------------------------------------------------
def test_inference_returns_a_prediction_for_every_image(prediction, image_files):
    assert isinstance(prediction, Prediction)
    n = len(image_files)
    assert prediction.depth.shape[0] == n
    assert prediction.conf.shape == prediction.depth.shape
    assert prediction.extrinsics.shape == (n, 3, 4)
    assert prediction.intrinsics.shape == (n, 3, 3)
    assert prediction.processed_images.shape == (*prediction.depth.shape, 3)


def test_the_prediction_is_numpy_not_torch(prediction):
    """Everything past ``OutputProcessor`` is numpy; the exporters index and
    ``np.round`` it without conversion."""
    for value in (
        prediction.depth,
        prediction.conf,
        prediction.extrinsics,
        prediction.intrinsics,
        prediction.processed_images,
    ):
        assert isinstance(value, np.ndarray)


def test_processed_images_are_displayable_uint8(prediction):
    images = prediction.processed_images
    assert images.dtype == np.uint8
    assert images.min() >= 0 and images.max() <= 255
    # Not a constant frame: the denormalisation would flatten it if it were wrong.
    assert images.std() > 1.0


def test_the_processed_resolution_honours_the_request(prediction):
    h, w = prediction.depth.shape[1:]
    assert max(h, w) == PROCESS_RES
    assert h % 14 == 0 and w % 14 == 0


def test_depth_and_confidence_keep_the_heads_guarantees(prediction):
    assert np.all(prediction.depth > 0)
    assert np.all(prediction.conf > 1.0)
    assert np.isfinite(prediction.depth).all()


def test_predicted_poses_are_rigid(prediction):
    r = prediction.extrinsics[:, :3, :3]
    np.testing.assert_allclose(
        r @ np.swapaxes(r, -1, -2), np.broadcast_to(np.eye(3), r.shape), atol=1e-5
    )


def test_inference_is_reproducible(tiny_api_model, image_files):
    a = tiny_api_model.inference(image_files, process_res=PROCESS_RES)
    b = tiny_api_model.inference(image_files, process_res=PROCESS_RES)
    np.testing.assert_array_equal(a.depth, b.depth)
    np.testing.assert_array_equal(a.extrinsics, b.extrinsics)


# ---------------------------------------------------------------------------
# export from the public entry point
# ---------------------------------------------------------------------------
def test_inference_writes_the_requested_export(tiny_api_model, image_files, tmp_path):
    tiny_api_model.inference(
        image_files,
        process_res=PROCESS_RES,
        export_dir=str(tmp_path),
        export_format="mini_npz",
    )
    data = np.load(_wait_for(tmp_path / "exports" / "mini_npz" / "results.npz"))
    assert data["depth"].shape[0] == len(image_files)


def test_glb_export_reaches_the_writer_with_its_options(
    tiny_api_model, image_files, tmp_path, monkeypatch
):
    """``inference`` assembles ``export_kwargs["glb"]`` from its own
    parameters. If that assembly broke, the CLI's GLB flags would be
    silently ignored."""
    from depth_anything_3.utils import export as export_mod

    seen = {}
    monkeypatch.setattr(export_mod, "export_to_glb", lambda p, d, **k: seen.update(k))

    tiny_api_model.inference(
        image_files,
        process_res=PROCESS_RES,
        export_dir=str(tmp_path),
        export_format="glb",
        num_max_points=321,
        show_cameras=False,
        conf_thresh_percentile=12.5,
    )

    assert seen == {"num_max_points": 321, "show_cameras": False, "conf_thresh_percentile": 12.5}


def test_requesting_a_gaussian_export_without_the_gaussian_branch_is_refused(
    tiny_api_model, image_files, tmp_path
):
    """``da3-small`` has no gaussian head at all, and ``inference`` guards the
    combination before spending a forward pass on it."""
    with pytest.raises(AssertionError, match="infer_gs"):
        tiny_api_model.inference(
            image_files, process_res=PROCESS_RES, export_dir=str(tmp_path), export_format="gs_ply"
        )


def test_colmap_export_requires_file_paths(tiny_api_model, image_files, tmp_path):
    """COLMAP writes an ``images.bin`` that references the originals by name,
    so an in-memory array cannot be exported."""
    arrays = [np.zeros((28, 28, 3), dtype=np.uint8)]
    with pytest.raises(AssertionError, match="image paths"):
        tiny_api_model.inference(
            arrays, process_res=PROCESS_RES, export_dir=str(tmp_path), export_format="colmap"
        )


def test_requested_feature_layers_come_back_in_aux(tiny_api_model, image_files):
    prediction = tiny_api_model.inference(
        image_files, process_res=PROCESS_RES, export_feat_layers=[5, 9]
    )
    assert set(prediction.aux) == {"feat_layer_5", "feat_layer_9"}
    for value in prediction.aux.values():
        assert value.shape[:3] == (
            len(image_files),
            *(d // 14 for d in prediction.depth.shape[1:]),
        )


# ---------------------------------------------------------------------------
# supplying camera parameters
# ---------------------------------------------------------------------------
def test_supplied_poses_are_returned_and_the_depth_is_rescaled(tiny_api_model, image_files):
    """With ``align_to_input_ext_scale`` the prediction adopts the *user's*
    poses and the depth is divided by the fitted scale, so the point cloud
    still lands where the cameras say it does."""
    from conftest import random_se3

    n = len(image_files)
    extrinsics = random_se3(n, seed=5).numpy().astype(np.float32)
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = 40.0

    free = tiny_api_model.inference(image_files, process_res=PROCESS_RES)
    aligned = tiny_api_model.inference(
        image_files,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        process_res=PROCESS_RES,
        align_to_input_ext_scale=True,
    )

    np.testing.assert_allclose(aligned.extrinsics, extrinsics[:, :3, :], rtol=1e-5)
    # The depth changed only by a single global factor.
    ratio = free.depth / aligned.depth
    assert ratio.std() / abs(ratio.mean()) < 1e-3


def test_supplied_camera_arrays_are_not_mutated(tiny_api_model, image_files):
    n = len(image_files)
    extrinsics = random_se3_float(n)
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    before = (extrinsics.copy(), intrinsics.copy())

    tiny_api_model.inference(
        image_files, extrinsics=extrinsics, intrinsics=intrinsics, process_res=PROCESS_RES
    )

    np.testing.assert_array_equal(extrinsics, before[0])
    np.testing.assert_array_equal(intrinsics, before[1])


def random_se3_float(n):
    from conftest import random_se3

    return random_se3(n, seed=2).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# the geometry-carrying stages, checked individually
# ---------------------------------------------------------------------------
def test_extrinsic_normalisation_rebases_onto_the_first_camera(tiny_api_model):
    """``_normalize_extrinsics`` moves the trajectory into the first camera's
    frame and scales it so the median camera distance is 1 -- that is what
    makes a learned scale-free model comparable to user poses."""
    from conftest import random_se3

    ext = random_se3(6, seed=8).float()[None]
    out = tiny_api_model._normalize_extrinsics(ext.clone())

    torch.testing.assert_close(out[0, 0], torch.eye(4), atol=1e-5, rtol=0)
    c2w = torch.linalg.inv(out[0].double())
    distances = c2w[:, :3, 3].norm(dim=-1)
    assert float(distances.median()) == pytest.approx(1.0, abs=1e-4)


def test_extrinsic_normalisation_preserves_relative_poses(tiny_api_model):
    """It may only rotate, translate and uniformly scale the trajectory."""
    from conftest import random_se3

    ext = random_se3(6, seed=9).float()[None]
    out = tiny_api_model._normalize_extrinsics(ext.clone())

    before = torch.linalg.inv(ext[0].double())[:, :3, 3]
    after = torch.linalg.inv(out[0].double())[:, :3, 3]
    pair_before = torch.cdist(before, before)
    pair_after = torch.cdist(after, after)
    ratio = pair_after[pair_before > 0] / pair_before[pair_before > 0]
    assert float(ratio.std()) < 1e-4


def test_extrinsic_normalisation_is_a_no_op_when_there_is_nothing_to_do(tiny_api_model):
    assert tiny_api_model._normalize_extrinsics(None) is None


def test_processed_images_invert_the_imagenet_normalisation(tiny_api_model):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original = np.random.default_rng(0).random((2, 3, 8, 10))
    normalised = torch.from_numpy(
        (original - mean[None, :, None, None]) / std[None, :, None, None]
    )

    prediction = tiny_api_model._add_processed_images(
        Prediction(depth=np.zeros((2, 8, 10)), is_metric=0), normalised
    )

    np.testing.assert_allclose(
        prediction.processed_images / 255.0, original.transpose(0, 2, 3, 1), atol=1 / 255
    )


def test_the_model_reports_the_device_its_parameters_live_on(tiny_api_model):
    assert tiny_api_model._get_model_device() == next(tiny_api_model.model.parameters()).device
