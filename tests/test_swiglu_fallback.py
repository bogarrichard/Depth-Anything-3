"""``model/dinov2/layers/swiglu_ffn.py``.

Used to wrap an xformers fused kernel with a pure-torch fallback for when
xformers was unavailable or mismatched the local torch/CUDA/Python build.
Removed: a GPU benchmark (RTX PRO 5000, torch 2.13/cu130/py3.13) showed the
xformers path silently degraded to its own eager Python SwiGLU on that stack
(xformers couldn't load its C++/CUDA extension) and was indistinguishable in
speed from the pure-torch implementation here -- while `torch.compile` on the
pure-torch implementation beat both by ~12%. ``SwiGLUFFNFused`` is now plain
pure-torch, with no fused-kernel dependency at all.

These tests pin what still matters: the checkpoint-visible parameter layout
and the hidden-width rounding rule (``vitg`` checkpoints were saved with
xformers' 2/3-rounded hidden width, so that arithmetic must not drift even
though nothing here calls xformers anymore).

Only ``vitg`` reaches this path (``dinov2.py`` picks ``swiglufused`` for that
backbone alone), which is exactly why it is the least likely thing to be
noticed if it breaks.
"""

import pytest
import torch
import torch.nn.functional as F

from depth_anything_3.model.dinov2.layers.swiglu_ffn import (
    SwiGLUFFN,
    SwiGLUFFNFused,
)

IN, HIDDEN, OUT = 32, 96, 32


def _expected_hidden(hidden: int) -> int:
    """The rounding SwiGLU checkpoints were saved with: two thirds of the
    requested width, rounded up to a multiple of 8."""
    return (int(hidden * 2 / 3) + 7) // 8 * 8


# ---------------------------------------------------------------------------
# the checkpoint-visible layout
# ---------------------------------------------------------------------------
def test_the_fused_wrapper_rounds_the_hidden_width():
    layer = SwiGLUFFNFused(in_features=IN, hidden_features=HIDDEN, out_features=OUT)
    hidden = _expected_hidden(HIDDEN)
    assert hidden == 64
    assert layer.w12.weight.shape == (2 * hidden, IN)
    assert layer.w3.weight.shape == (OUT, hidden)


@pytest.mark.parametrize("hidden", [96, 100, 128, 6144])
def test_the_rounding_rule_holds_for_every_width(hidden):
    layer = SwiGLUFFNFused(in_features=IN, hidden_features=hidden, out_features=OUT)
    assert layer.w12.weight.shape[0] == 2 * _expected_hidden(hidden)


def test_the_parameter_names_are_the_ones_checkpoints_use():
    """A rename here turns every ``vitg`` weight into a "missing key" that
    ``load_state_dict(strict=False)`` reports to a log and shrugs off."""
    layer = SwiGLUFFNFused(in_features=IN, hidden_features=HIDDEN, out_features=OUT)
    assert set(layer.state_dict()) == {"w12.weight", "w12.bias", "w3.weight", "w3.bias"}


def test_the_two_projections_are_packed_into_one_matrix():
    """``w12`` holds the gate and the value branch stacked, in that order.
    Splitting them into separate ``w1``/``w2`` tensors -- or swapping the
    halves -- would load without complaint and compute something else."""
    layer = SwiGLUFFN(in_features=IN, hidden_features=HIDDEN, out_features=OUT)
    x = torch.randn(4, IN)

    gate, value = layer.w12(x).chunk(2, dim=-1)
    expected = layer.w3(F.silu(gate) * value)

    torch.testing.assert_close(layer(x), expected)


def test_the_activation_is_silu_on_the_first_half_only():
    """Applying silu to the second half instead is a plausible typo that
    keeps every shape intact."""
    layer = SwiGLUFFN(in_features=IN, hidden_features=HIDDEN, out_features=OUT)
    x = torch.randn(4, IN)
    gate, value = layer.w12(x).chunk(2, dim=-1)
    swapped = layer.w3(F.silu(value) * gate)
    assert not torch.allclose(layer(x), swapped)


def test_defaults_follow_the_input_width():
    layer = SwiGLUFFN(in_features=IN)
    assert layer.w12.weight.shape == (2 * IN, IN)
    assert layer.w3.weight.shape == (IN, IN)


def test_a_checkpoint_saved_with_the_rounded_width_loads_directly():
    """``SwiGLUFFNFused`` rounds its hidden width; a plain ``SwiGLUFFN`` built
    with that already-rounded width must accept the same state dict -- this is
    what let old xformers-fused checkpoints load into the pure-torch class
    without a shape mismatch."""
    hidden = _expected_hidden(HIDDEN)
    fused = SwiGLUFFNFused(in_features=IN, hidden_features=HIDDEN, out_features=OUT)
    plain = SwiGLUFFN(in_features=IN, hidden_features=hidden, out_features=OUT)
    plain.load_state_dict(fused.state_dict())


# ---------------------------------------------------------------------------
# which backbones reach it
# ---------------------------------------------------------------------------
def test_only_the_giant_backbone_selects_swiglu():
    """``dinov2.py`` picks ``swiglufused`` for ``vitg`` and ``mlp`` for the
    rest. Widening that would change the parameter layout of every other
    preset's checkpoints."""
    from depth_anything_3.model.dinov2 import vision_transformer as vit
    from depth_anything_3.model.dinov2.layers import Mlp

    giant = vit.vit_giant2(patch_size=14, img_size=518, depth=1, ffn_layer="swiglufused")
    assert isinstance(giant.blocks[0].mlp, SwiGLUFFNFused)

    small = vit.vit_small(patch_size=14, img_size=518, depth=1, ffn_layer="mlp")
    assert isinstance(small.blocks[0].mlp, Mlp)


@pytest.mark.parametrize(
    "name,expected", [("vits", "mlp"), ("vitb", "mlp"), ("vitl", "mlp"), ("vitg", "swiglufused")]
)
def test_the_backbone_selects_the_right_ffn_for_each_size(monkeypatch, name, expected):
    """The selection itself, observed through the factory it calls -- so no
    1.1B-parameter model has to be allocated to find out."""
    from depth_anything_3.model.dinov2 import dinov2 as dinov2_module

    seen = {}

    def _record(**kwargs):
        seen.update(kwargs)
        return torch.nn.Identity()

    for factory in ("vit_small", "vit_base", "vit_large", "vit_giant2"):
        monkeypatch.setattr(dinov2_module, factory, _record)

    dinov2_module.DinoV2(name=name, out_layers=[0])

    assert seen["ffn_layer"] == expected
    assert seen["patch_size"] == 14


def test_the_backbone_rejects_a_size_it_has_no_factory_for():
    from depth_anything_3.model.dinov2.dinov2 import DinoV2

    with pytest.raises(AssertionError):
        DinoV2(name="vitxl", out_layers=[0])
