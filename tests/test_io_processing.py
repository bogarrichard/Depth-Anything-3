"""``InputProcessor`` and ``OutputProcessor``: the boundary of the package.

Everything a user hands in crosses ``InputProcessor`` and everything they get
back crosses ``OutputProcessor``. Both are pure plumbing -- resizing,
normalising, squeezing batch dimensions -- and both are places where an
off-by-one or a transposed axis produces perfectly plausible output.

The intrinsics tests are the important ones: an image resize that does not
carry the calibration with it produces a reconstruction that looks right and
is quietly the wrong scale. They are written as invariants (the *normalised*
focal length and principal point are what a resize must preserve) rather than
by copying the scaling expression out of the implementation.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.output_processor import OutputProcessor

PATCH = InputProcessor.PATCH_SIZE
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
METHODS = ["upper_bound_resize", "lower_bound_resize", "upper_bound_crop", "lower_bound_crop"]


@pytest.fixture(scope="module")
def processor():
    return InputProcessor()


def _image(w: int, h: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _pinhole(w: int, h: int) -> np.ndarray:
    return np.array(
        [[0.9 * w, 0.0, 0.5 * w], [0.0, 1.1 * h, 0.45 * h], [0.0, 0.0, 1.0]], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# InputProcessor: shapes
# ---------------------------------------------------------------------------
def test_the_batch_tensor_is_channels_first_without_a_batch_dimension(processor):
    """(N, 3, H, W) -- ``api._prepare_model_inputs`` adds the leading batch
    axis itself with ``[None]``. The class docstring says (1, N, 3, H, W);
    it is the docstring that is wrong."""
    images = [_image(80, 64, seed=i) for i in range(3)]
    batch, ext, ixt = processor(images, None, None, 56, "upper_bound_resize")
    assert batch.shape[0] == 3 and batch.shape[1] == 3 and batch.ndim == 4
    assert batch.dtype == torch.float32
    assert ext is None and ixt is None


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("size", [(80, 64), (37, 121), (14, 14)])
def test_every_method_yields_patch_aligned_sizes(processor, method, size):
    """The backbone divides H and W by 14; anything else raises deep inside
    the patch embedding."""
    batch, *_ = processor([_image(*size)], None, None, 56, method)
    _, _, h, w = batch.shape
    assert h % PATCH == 0 and w % PATCH == 0 and h > 0 and w > 0


def test_upper_bound_resize_caps_the_longest_side(processor):
    batch, *_ = processor([_image(100, 40)], None, None, 56, "upper_bound_resize")
    assert max(batch.shape[-2:]) == 56


def test_lower_bound_resize_lifts_the_shortest_side(processor):
    batch, *_ = processor([_image(100, 40)], None, None, 56, "lower_bound_resize")
    assert min(batch.shape[-2:]) == 56


def test_an_unsupported_method_is_rejected(processor):
    with pytest.raises(ValueError, match="middle_bound_wobble"):
        processor([_image(28, 28)], None, None, 28, "middle_bound_wobble")


def test_an_unsupported_image_type_is_rejected(processor):
    with pytest.raises(ValueError, match="Unsupported image type"):
        processor([12345], None, None, 28, "upper_bound_resize")


# ---------------------------------------------------------------------------
# InputProcessor: pixel values
# ---------------------------------------------------------------------------
def test_normalisation_is_imagenet_and_reversible(processor):
    """Chosen so no resizing happens (56 is already the longest side and both
    sides are multiples of 14), which makes the pixel values exactly
    predictable."""
    image = _image(56, 42)
    batch, *_ = processor([image], None, None, 56, "upper_bound_resize")

    recovered = batch[0].permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
    np.testing.assert_allclose(recovered * 255, image, atol=0.51)


@pytest.mark.parametrize("kind", ["ndarray", "pil", "path"])
def test_the_three_accepted_input_kinds_agree(processor, tmp_path, kind):
    array = _image(56, 42, seed=4)
    if kind == "ndarray":
        item = array
    elif kind == "pil":
        item = Image.fromarray(array)
    else:
        path = tmp_path / "one.png"
        Image.fromarray(array).save(path)
        item = str(path)

    batch, *_ = processor([item], None, None, 56, "upper_bound_resize")
    reference, *_ = processor([array], None, None, 56, "upper_bound_resize")
    torch.testing.assert_close(batch, reference)


def test_output_order_matches_input_order(processor):
    """Images are processed by a worker pool. If the pool ever stopped
    preserving order, depth maps would silently swap between images."""
    images = [np.full((42, 56, 3), fill_value=v, dtype=np.uint8) for v in (10, 120, 250)]
    batch, *_ = processor(images, None, None, 56, "upper_bound_resize", num_workers=8)
    means = [float(batch[i].mean()) for i in range(3)]
    assert means == sorted(means)


def test_sequential_and_parallel_paths_agree(processor):
    images = [_image(80, 64, seed=i) for i in range(4)]
    parallel, *_ = processor(images, None, None, 56, "upper_bound_resize", num_workers=4)
    sequential, *_ = processor(images, None, None, 56, "upper_bound_resize", sequential=True)
    torch.testing.assert_close(parallel, sequential)


def test_differently_sized_images_are_unified_by_centre_crop(processor):
    images = [_image(84, 84), _image(56, 42)]
    batch, *_ = processor(images, None, None, 84, "upper_bound_resize")
    assert batch.shape[0] == 2
    # Everything is cropped down to the smallest processed size in the batch.
    assert batch.shape[-2] % PATCH == 0 and batch.shape[-1] % PATCH == 0


# ---------------------------------------------------------------------------
# InputProcessor: camera parameters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["upper_bound_resize", "lower_bound_resize"])
def test_a_resize_preserves_the_normalised_intrinsics(processor, method):
    """Scaling an image by s must scale fx, fy, cx, cy by the same s -- which
    is the same as saying fx/W and cx/W are invariant. That invariant, not
    the scaling expression, is what the geometry depends on."""
    w, h = 100, 64
    k = _pinhole(w, h)
    batch, _, out_k = processor([_image(w, h)], None, k[None], 56, method)
    out_k = out_k[0].numpy()
    _, _, new_h, new_w = batch.shape

    np.testing.assert_allclose(out_k[0, 0] / new_w, k[0, 0] / w, rtol=1e-3)
    np.testing.assert_allclose(out_k[0, 2] / new_w, k[0, 2] / w, rtol=1e-3)
    np.testing.assert_allclose(out_k[1, 1] / new_h, k[1, 1] / h, rtol=1e-3)
    np.testing.assert_allclose(out_k[1, 2] / new_h, k[1, 2] / h, rtol=1e-3)
    assert out_k[2, 2] == 1.0


def test_a_centre_crop_moves_the_principal_point_and_leaves_the_focal_alone(processor):
    """The unit rule behind the ``*_crop`` methods: cropping changes where
    the optical axis sits in the image, never how wide the lens is."""
    k = _pinhole(120, 90)
    cropped = processor._crop_ixt(k, orig_w=120, orig_h=90, w=100, h=70)
    assert cropped[0, 0] == k[0, 0] and cropped[1, 1] == k[1, 1]
    assert cropped[0, 2] == k[0, 2] - 10
    assert cropped[1, 2] == k[1, 2] - 10


def test_extrinsics_pass_through_untouched(processor):
    from conftest import random_se3

    ext = random_se3(2).numpy()
    _, out_ext, _ = processor(
        [_image(56, 42), _image(56, 42, seed=1)], ext, None, 56, "upper_bound_resize"
    )
    np.testing.assert_allclose(out_ext.numpy(), ext.astype(np.float32), atol=1e-6)


def test_camera_arrays_must_match_the_image_count(processor):
    from conftest import random_se3

    with pytest.raises(ValueError, match="extrinsics"):
        processor([_image(28, 28)], random_se3(3).numpy(), None, 28, "upper_bound_resize")
    with pytest.raises(ValueError, match="intrinsics"):
        processor([_image(28, 28)], None, np.tile(np.eye(3), (2, 1, 1)), 28, "upper_bound_resize")


def test_the_processor_does_not_mutate_the_callers_arrays(processor):
    """``api.inference`` defensively copies before calling; the processor
    itself must not need that."""
    k = _pinhole(56, 42)[None].copy()
    original = k.copy()
    processor([_image(56, 42)], None, k, 56, "upper_bound_resize")
    np.testing.assert_array_equal(k, original)


# ---------------------------------------------------------------------------
# OutputProcessor
# ---------------------------------------------------------------------------
def _model_output(**overrides):
    out = {
        "depth": torch.rand(1, 3, 8, 10),
        "depth_conf": torch.rand(1, 3, 8, 10) + 1,
        "extrinsics": torch.eye(4).expand(1, 3, 4, 4),
        "intrinsics": torch.eye(3).expand(1, 3, 3, 3),
    }
    out.update(overrides)
    return {k: v for k, v in out.items() if v is not None}


def test_the_leading_batch_dimension_is_dropped():
    prediction = OutputProcessor()(_model_output())
    assert isinstance(prediction, Prediction)
    assert prediction.depth.shape == (3, 8, 10)
    assert prediction.conf.shape == (3, 8, 10)
    assert prediction.extrinsics.shape == (3, 4, 4)
    assert prediction.intrinsics.shape == (3, 3, 3)


def test_everything_comes_back_as_numpy():
    prediction = OutputProcessor()(_model_output())
    for value in (prediction.depth, prediction.conf, prediction.extrinsics, prediction.intrinsics):
        assert isinstance(value, np.ndarray)


def test_values_are_carried_across_unchanged():
    out = _model_output()
    prediction = OutputProcessor()(out)
    np.testing.assert_array_equal(prediction.depth, out["depth"][0].numpy())
    np.testing.assert_array_equal(prediction.extrinsics, out["extrinsics"][0].numpy())


def test_optional_fields_become_none_when_absent():
    prediction = OutputProcessor()(
        _model_output(depth_conf=None, extrinsics=None, intrinsics=None)
    )
    assert prediction.conf is None
    assert prediction.extrinsics is None
    assert prediction.intrinsics is None
    assert prediction.sky is None


def test_the_sky_head_is_thresholded_into_a_boolean_mask():
    sky = torch.tensor([[[0.0, 0.49], [0.5, 1.0]]])[None]
    prediction = OutputProcessor()(_model_output(sky=sky))
    assert prediction.sky.dtype == np.bool_
    assert prediction.sky.tolist() == [[[False, False], [True, True]]]


def test_auxiliary_features_are_converted_and_debatched():
    aux = {"feat_layer_3": torch.rand(1, 3, 4, 5, 6)}
    prediction = OutputProcessor()(_model_output(aux=aux))
    assert set(prediction.aux) == {"feat_layer_3"}
    assert prediction.aux["feat_layer_3"].shape == (3, 4, 5, 6)
    assert isinstance(prediction.aux["feat_layer_3"], np.ndarray)


def test_is_metric_is_falsy_for_a_plain_prediction():
    """``Prediction.is_metric`` is annotated ``int`` and read as a boolean by
    the gaussian exporter. The extractor reads it with ``dict.get``, so a model
    output that never sets the key yields the ``0`` default and keeps the
    annotation honest -- not merely something falsy."""
    assert OutputProcessor()(_model_output()).is_metric == 0


def test_is_metric_survives_when_the_nested_model_sets_it():
    from addict import Dict as AddictDict

    out = AddictDict(_model_output())
    out.is_metric = 1
    assert OutputProcessor()(out).is_metric == 1
