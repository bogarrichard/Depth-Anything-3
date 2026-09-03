"""Metric-depth alignment primitives in ``utils/alignment.py``.

These five small functions decide how the nested (metric) model's depth is
fused into the main prediction. They are pure and cheap, and every one of
them has a sign, a polarity or a threshold that is easy to invert without
anything downstream crashing -- the reconstruction just comes out at the
wrong scale. Each is checked against its definition, and the mask polarity
in particular is asserted rather than assumed.
"""

import numpy as np
import pytest
import torch

from depth_anything_3.utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)


# ---------------------------------------------------------------------------
# least_squares_scale_scalar
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scale_is_exact_when_the_signals_are_proportional(dtype):
    b = torch.randn(64, dtype=dtype)
    scale = least_squares_scale_scalar(3.5 * b, b)
    assert scale.dtype == dtype and scale.ndim == 0
    torch.testing.assert_close(scale, torch.tensor(3.5, dtype=dtype))


def test_scale_minimises_the_squared_error():
    """The defining property, checked against numpy's own least squares."""
    a = torch.randn(128, dtype=torch.float64)
    b = torch.randn(128, dtype=torch.float64)
    s = float(least_squares_scale_scalar(a, b))
    expected, *_ = np.linalg.lstsq(b.numpy()[:, None], a.numpy(), rcond=None)
    assert s == pytest.approx(float(expected[0]), rel=1e-12)
    # ...and no nearby scale does better.
    residual = float(((a - s * b) ** 2).sum())
    for delta in (-1e-3, 1e-3):
        assert float(((a - (s + delta) * b) ** 2).sum()) > residual


def test_scale_works_on_multidimensional_input():
    a = torch.randn(4, 5, 6, dtype=torch.float64)
    torch.testing.assert_close(
        least_squares_scale_scalar(2.0 * a, a), torch.tensor(2.0, dtype=torch.float64)
    )


def test_scale_does_not_divide_by_zero():
    zero = torch.zeros(8, dtype=torch.float64)
    assert torch.isfinite(least_squares_scale_scalar(torch.ones(8, dtype=torch.float64), zero))


def test_scale_rejects_mismatched_and_non_float_input():
    with pytest.raises(ValueError):
        least_squares_scale_scalar(torch.zeros(4), torch.zeros(5))
    with pytest.raises(TypeError):
        least_squares_scale_scalar(
            torch.zeros(4, dtype=torch.int64), torch.zeros(4, dtype=torch.int64)
        )


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def test_compute_sky_mask_returns_the_NON_sky_mask():
    """Named ``compute_sky_mask``, documented as "True indicates non-sky".
    Callers spell it ``non_sky_mask``. Inverting it would push the sky depth
    onto the ground instead of the other way round."""
    sky_probability = torch.tensor([0.0, 0.29, 0.31, 1.0])
    mask = compute_sky_mask(sky_probability, threshold=0.3)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, True, False, False]


def test_compute_sky_mask_threshold_is_exclusive_on_the_upper_side():
    exactly_at = torch.tensor([0.3])
    assert compute_sky_mask(exactly_at, threshold=0.3).tolist() == [False]


def test_alignment_mask_requires_every_condition():
    """Four independent gates; each one alone must be able to veto a pixel."""
    ones = torch.ones(4)
    base = dict(
        depth_conf=ones * 2.0,
        non_sky_mask=torch.ones(4, dtype=torch.bool),
        depth=ones,
        metric_depth=ones,
        median_conf=torch.tensor(1.0),
    )
    assert compute_alignment_mask(**base).tolist() == [True] * 4

    for key, veto in [
        ("depth_conf", torch.tensor([0.0, 2.0, 2.0, 2.0])),
        ("non_sky_mask", torch.tensor([False, True, True, True])),
        ("depth", torch.tensor([0.0, 1.0, 1.0, 1.0])),
        ("metric_depth", torch.tensor([0.0, 1.0, 1.0, 1.0])),
    ]:
        mask = compute_alignment_mask(**{**base, key: veto})
        assert mask.tolist() == [False, True, True, True], key


def test_alignment_mask_thresholds_are_strict_inequalities_on_depth():
    """A depth of exactly ``min_depth_threshold`` is rejected, so a zero-depth
    pixel can never sneak into the least-squares fit."""
    args = dict(
        depth_conf=torch.ones(1),
        non_sky_mask=torch.ones(1, dtype=torch.bool),
        depth=torch.tensor([1e-3]),
        metric_depth=torch.tensor([1.0]),
        median_conf=torch.tensor(1.0),
    )
    assert compute_alignment_mask(**args).tolist() == [False]
    args["depth"] = torch.tensor([1.1e-3])
    assert compute_alignment_mask(**args).tolist() == [True]


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def test_small_tensors_are_returned_untouched():
    x = torch.randn(10, 3)
    assert sample_tensor_for_quantile(x, max_samples=100) is x


def test_large_tensors_are_subsampled_without_replacement():
    x = torch.arange(1000.0)
    out = sample_tensor_for_quantile(x, max_samples=50)
    assert out.shape == (50,)
    assert len(set(out.tolist())) == 50, "randperm must not repeat elements"
    assert set(out.tolist()) <= set(x.tolist())


def test_subsampling_preserves_the_distribution_well_enough_for_a_quantile():
    """It exists so ``torch.quantile`` stays within its 16M element limit; the
    sample has to stand in for the whole tensor."""
    torch.manual_seed(0)
    x = torch.randn(500_000)
    sampled = sample_tensor_for_quantile(x, max_samples=100_000)
    assert abs(float(x.quantile(0.99)) - float(sampled.quantile(0.99))) < 0.05


# ---------------------------------------------------------------------------
# metric scaling and sky filling
# ---------------------------------------------------------------------------
def test_metric_scaling_uses_the_mean_focal_length():
    depth = torch.ones(1, 2, 4, 5)
    k = torch.zeros(1, 2, 3, 3)
    k[0, 0, 0, 0], k[0, 0, 1, 1] = 100.0, 200.0  # mean 150
    k[0, 1, 0, 0], k[0, 1, 1, 1] = 600.0, 600.0  # mean 600

    out = apply_metric_scaling(depth, k, scale_factor=300.0)

    torch.testing.assert_close(out[0, 0], torch.full((4, 5), 0.5))
    torch.testing.assert_close(out[0, 1], torch.full((4, 5), 2.0))


def test_metric_scaling_is_linear_in_depth():
    depth = torch.rand(1, 1, 3, 3) + 0.1
    k = torch.eye(3).expand(1, 1, 3, 3).contiguous()
    torch.testing.assert_close(
        apply_metric_scaling(2 * depth, k), 2 * apply_metric_scaling(depth, k)
    )


def test_sky_regions_are_filled_and_confidence_pinned():
    depth = torch.arange(4.0)
    conf = torch.zeros(4)
    non_sky = torch.tensor([True, True, False, False])

    new_depth, new_conf = set_sky_regions_to_max_depth(depth, conf, non_sky, max_depth=99.0)

    assert new_depth.tolist() == [0.0, 1.0, 99.0, 99.0]
    assert new_conf.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_sky_fill_does_not_mutate_its_input():
    depth = torch.arange(4.0)
    conf = torch.zeros(4)
    set_sky_regions_to_max_depth(depth, conf, torch.tensor([True, False, True, False]), 99.0)
    assert depth.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert conf.tolist() == [0.0] * 4


def test_sky_fill_tolerates_a_missing_confidence_map():
    """``da3.py`` calls it with ``depth_conf=None`` for the mono-sky branch."""
    depth, conf = set_sky_regions_to_max_depth(
        torch.arange(4.0), None, torch.tensor([True, True, False, False]), 7.0
    )
    assert conf is None
    assert depth.tolist() == [0.0, 1.0, 7.0, 7.0]
