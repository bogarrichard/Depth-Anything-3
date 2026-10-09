"""Torch uint8 image input for ``InputProcessor`` and ``DepthAnything3.inference``.

The NumPy/PIL path (PIL + ``cv2.resize``) is the reference: for the same sizes and
``process_res_method`` the tensor path must give the same shapes and the same adjusted
intrinsics exactly, and pixel values within a small tolerance (measured: at most one
8-bit level, i.e. <= 1/255 of the range, on up- and downscaling; the cv2 rounding
differs in the last bit).
"""

import numpy as np
import pytest
import torch
from _oracles import preprocess_image_oracle

from depth_anything_3.utils.io.input_processor import InputProcessor

METHODS = ["upper_bound_resize", "lower_bound_resize", "upper_bound_crop", "lower_bound_crop"]
# (w, h): downscale, upscale, odd sizes, portrait, already aligned
SIZES = [(640, 480), (100, 77), (333, 251), (251, 333), (56, 42), (31, 17)]
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@pytest.fixture(scope="module")
def proc():
    return InputProcessor()


def _img(w, h, seed=0):
    rng = np.random.default_rng(seed)
    # smooth-ish content plus noise, like a photo
    base = np.cumsum(np.cumsum(rng.normal(0, 1, (h, w, 3)), 0), 1) * 0.3 + 128
    return np.clip(base + rng.normal(0, 8, (h, w, 3)), 0, 255).astype(np.uint8)


def _K(w, h):
    return np.array([[0.9 * w, 0, 0.5 * w], [0, 1.1 * h, 0.45 * h], [0, 0, 1.0]])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("res", [56, 140])
def test_shapes_intrinsics_and_pixels_match_the_numpy_path(proc, method, size, res):
    w, h = size
    a = _img(w, h)
    K = [_K(w, h)]
    old, _, k_old = proc([a], None, K, res, method, num_workers=1)
    new, _, k_new = proc([torch.from_numpy(a)], None, K, res, method)
    assert new.shape == old.shape and new.dtype == torch.float32
    np.testing.assert_array_equal(k_new.numpy(), k_old.numpy())
    err_levels = ((new - old).abs() * STD * 255).max().item()
    assert err_levels <= 1.0 + 1e-3  # at most one 8-bit level


@pytest.mark.parametrize("layout", ["nhwc", "nchw", "list_hwc", "list_chw"])
def test_all_accepted_layouts_agree(proc, layout):
    arrs = [_img(80, 60, seed=i) for i in range(3)]
    nhwc = torch.from_numpy(np.stack(arrs))
    x = {
        "nhwc": nhwc,
        "nchw": nhwc.permute(0, 3, 1, 2).contiguous(),
        "list_hwc": list(nhwc),
        "list_chw": list(nhwc.permute(0, 3, 1, 2)),
    }[layout]
    ref, *_ = proc(nhwc, None, None, 56, "upper_bound_resize")
    out, *_ = proc(x, None, None, 56, "upper_bound_resize")
    torch.testing.assert_close(out, ref)
    assert out.shape[0] == 3


def test_differently_sized_tensors_are_centre_cropped_with_intrinsics(proc):
    a, b = _img(112, 84), _img(84, 112, seed=1)
    Ks = [_K(112, 84), _K(84, 112)]
    old, _, ko = proc([a, b], None, Ks, 56, "upper_bound_crop", num_workers=1)
    new, _, kn = proc([torch.from_numpy(a), torch.from_numpy(b)], None, Ks, 56, "upper_bound_crop")
    assert new.shape == old.shape
    np.testing.assert_array_equal(kn.numpy(), ko.numpy())


def test_extrinsics_pass_through_and_inputs_are_not_mutated(proc):
    t = torch.from_numpy(_img(70, 50))
    before = t.clone()
    E = np.eye(4)[None] * 1.0
    _, e, _ = proc([t], E, None, 56, "upper_bound_resize")
    np.testing.assert_array_equal(e.numpy(), E)
    assert torch.equal(t, before)


def test_device_is_preserved():
    t = torch.zeros(2, 40, 56, 3, dtype=torch.uint8, device="meta")
    out, *_ = InputProcessor()(t, None, None, 28, "upper_bound_resize")
    assert out.device.type == "meta" and out.shape == (2, 3, 14, 28)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_tensor_stays_on_the_gpu_and_matches_cpu(proc):
    a = torch.from_numpy(_img(160, 120))
    cpu, *_ = proc([a], None, None, 56, "upper_bound_resize")
    gpu, *_ = proc([a.cuda()], None, None, 56, "upper_bound_resize")
    assert gpu.is_cuda
    assert ((gpu.cpu() - cpu).abs() * STD * 255).max() <= 1.0 + 1e-3


@pytest.mark.parametrize(
    "bad, match",
    [
        (torch.zeros(2, 20, 20, 3), "uint8"),
        (torch.zeros(2, 20, 20, 4, dtype=torch.uint8), "3 channels"),
        (torch.zeros(20, 20, dtype=torch.uint8), "(N,H,W,3)"),
        (torch.zeros(3, 20, 3, dtype=torch.uint8), "Ambiguous"),
        (torch.zeros(0, 20, 20, 3, dtype=torch.uint8), "(N,H,W,3)"),
        ([torch.zeros(20, 20, 3, dtype=torch.uint8), np.zeros((20, 20, 3), np.uint8)], "mix"),
    ],
)
def test_bad_tensor_input_is_rejected(proc, bad, match):
    with pytest.raises(ValueError, match=match):
        proc(bad, None, None, 28, "upper_bound_resize")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("size", SIZES)
def test_both_paths_match_an_independent_pil_cv2_reference(proc, method, size):
    """The numpy path must equal the longhand cv2 reference (it is the unchanged old
    code), and the tensor path must be within one 8-bit level of it."""
    w, h = size
    a = _img(w, h, seed=3)
    ref, (sx, sy, cx, cy) = preprocess_image_oracle(a, 56, method)
    K = _K(w, h)
    k_ref = K.copy()
    k_ref[0] *= sx
    k_ref[1] *= sy
    k_ref[0, 2] -= cx
    k_ref[1, 2] -= cy

    old, _, k_old = proc([a], None, [K], 56, method, num_workers=1)
    np.testing.assert_allclose(old[0].numpy(), ref, atol=1e-5)
    np.testing.assert_allclose(k_old[0].numpy(), k_ref, atol=1e-4)

    new, _, k_new = proc([torch.from_numpy(a)], None, [K], 56, method)
    assert new.shape[1:] == ref.shape
    assert ((new[0].numpy() - ref) * STD[0].numpy() * 255).max() <= 1.0 + 1e-3
    np.testing.assert_allclose(k_new[0].numpy(), k_ref, atol=1e-4)


@pytest.mark.parametrize("n_in,n_out", [(480, 378), (640, 504), (100, 7), (17, 16), (5, 5)])
def test_area_weights_rows_sum_to_one(n_in, n_out):
    w = InputProcessor._area_weights(n_in, n_out, torch.device("cpu"))
    assert w.shape == (n_out, n_in)
    torch.testing.assert_close(w.sum(1), torch.ones(n_out))
    assert (w >= 0).all()


def test_area_weights_identity_at_scale_one():
    w = InputProcessor._area_weights(9, 9, torch.device("cpu"))
    torch.testing.assert_close(w, torch.eye(9))


def test_area_weights_integer_scale_is_block_averaging():
    w = InputProcessor._area_weights(12, 4, torch.device("cpu"))
    x = torch.arange(12, dtype=torch.float32)
    torch.testing.assert_close(w @ x, x.view(4, 3).mean(1))


def test_area_weights_are_cached():
    cpu = torch.device("cpu")
    assert InputProcessor._area_weights(30, 20, cpu) is InputProcessor._area_weights(30, 20, cpu)


@pytest.mark.parametrize(
    "src,dst", [((480, 640), (378, 504)), ((77, 100), (60, 41)), ((50, 50), (7, 49))]
)
def test_resize_tensor_downscale_matches_cv2_inter_area(src, dst):
    cv2 = pytest.importorskip("cv2")
    a = _img(src[1], src[0], seed=5)
    ref = cv2.resize(a, (dst[1], dst[0]), interpolation=cv2.INTER_AREA)
    out = InputProcessor._resize_tensor(
        torch.from_numpy(a).permute(2, 0, 1), dst[1], dst[0], False
    )
    assert out.dtype == torch.uint8
    diff = out.permute(1, 2, 0).numpy().astype(int) - ref.astype(int)
    assert np.abs(diff).max() <= 1


@pytest.mark.parametrize("src,dst", [((42, 56), (84, 112)), ((30, 41), (77, 53))])
def test_resize_tensor_upscale_matches_cv2_inter_cubic(src, dst):
    cv2 = pytest.importorskip("cv2")
    a = _img(src[1], src[0], seed=6)
    ref = cv2.resize(a, (dst[1], dst[0]), interpolation=cv2.INTER_CUBIC)
    out = InputProcessor._resize_tensor(torch.from_numpy(a).permute(2, 0, 1), dst[1], dst[0], True)
    diff = out.permute(1, 2, 0).numpy().astype(int) - ref.astype(int)
    assert np.abs(diff).max() <= 1


def test_nchw_batch_with_height_three_is_read_as_nchw():
    x = torch.randint(0, 255, (2, 3, 3, 40), dtype=torch.uint8)
    items = InputProcessor._as_tensor_list(x)
    assert len(items) == 2 and all(t.shape == (3, 3, 40) for t in items)
    assert torch.equal(items[1], x[1])


def test_nhwc_batch_with_width_or_height_three_is_read_as_nhwc():
    x = torch.randint(0, 255, (2, 40, 3, 3), dtype=torch.uint8)  # H=40, W=3
    items = InputProcessor._as_tensor_list(x)
    assert all(t.shape == (3, 40, 3) for t in items)
    assert torch.equal(items[0], x[0].permute(2, 0, 1))


@pytest.mark.parametrize("shape", [(2, 3, 40, 3), (2, 3, 3, 3)])
def test_batch_with_three_on_both_ends_is_ambiguous(shape):
    with pytest.raises(ValueError, match="Ambiguous"):
        InputProcessor._as_tensor_list(torch.zeros(shape, dtype=torch.uint8))


def test_single_3d_tensor_in_each_layout(proc):
    a = _img(60, 40, seed=2)
    t = torch.from_numpy(a)
    ref, *_ = proc([a], None, None, 56, "upper_bound_resize")
    for single in (t, t.permute(2, 0, 1).contiguous()):
        out, *_ = proc(single, None, None, 56, "upper_bound_resize")
        assert out.shape == ref.shape
        assert ((out - ref).abs() * STD * 255).max() <= 1.0 + 1e-3


def test_tensors_on_different_devices_are_rejected(proc):
    a = torch.zeros(20, 20, 3, dtype=torch.uint8)
    b = torch.zeros(20, 20, 3, dtype=torch.uint8, device="meta")
    with pytest.raises(ValueError, match="one device"):
        proc([a, b], None, None, 28, "upper_bound_resize")


def test_inference_accepts_tensors_and_agrees_with_numpy(tiny_api_model):
    arrs = [_img(80, 60, seed=i) for i in range(2)]
    old = tiny_api_model.inference(arrs, process_res=56)
    new = tiny_api_model.inference(torch.from_numpy(np.stack(arrs)), process_res=56)
    assert new.depth.shape == old.depth.shape
    assert new.processed_images.shape == old.processed_images.shape
    assert (
        np.abs(new.processed_images.astype(int) - old.processed_images.astype(int)).max() <= 2
    )  # 1 level + uint8 truncation on denormalise
    np.testing.assert_allclose(
        new.depth, old.depth, rtol=0.05, atol=0.05 * np.abs(old.depth).max()
    )
