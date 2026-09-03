"""Checkpoint key translation in ``utils/model_loading.py``.

This module is a table of ``str.replace`` calls that maps the names in the
released checkpoints onto the names in this code. It is invisible until it is
wrong, and when it is wrong ``load_state_dict(strict=False)`` reports the
mismatch to a logger and carries on with **randomly initialised weights** --
the model still runs, still returns plausible-looking depth, and is simply
useless.

So the tests here pin every documented rename individually, and then check
the table end to end against the key set of a real network.
"""

import pytest
import torch
import torch.nn as nn

from depth_anything_3.utils.model_loading import (
    convert_general_state_dict,
    convert_metric_state_dict,
    load_pretrained_weights,
)

# (legacy key, expected key). Each entry is one documented rename.
RENAMES = [
    ("module.head.weight", "model.head.weight"),
    ("model.net.blocks.0.weight", "model.backbone.blocks.0.weight"),
    ("model.backbone.pretrained.camera_token_extra", "model.backbone.pretrained.camera_token"),
    ("model.all_heads.camera_cond_head.fc.weight", "model.cam_enc.fc.weight"),
    ("model.all_heads.camera_head.fc.weight", "model.cam_dec.fc.weight"),
    ("model.cam_dec.more_mlps.0.weight", "model.cam_dec.backbone.0.weight"),
    ("model.cam_dec.fc_rot.weight", "model.cam_dec.fc_qvec.weight"),
    ("model.all_heads.head.scratch.weight", "model.head.scratch.weight"),
    (
        "model.head.output_conv2_additional.sky_mask.weight",
        "model.head.sky_output_conv2.weight",
    ),
    ("model.head.output_conv2_aux.weight", "model.head.output_conv2_aux.weight"),
    ("model.head.output_conv2_ray.weight", "model.head.output_conv2_aux.weight"),
    ("model.gaussian_param_head.conv.weight", "model.gs_head.conv.weight"),
]


def _tensor(i: int) -> torch.Tensor:
    return torch.full((2, 2), float(i))


@pytest.mark.parametrize("legacy,expected", RENAMES)
def test_each_documented_rename(legacy, expected):
    converted = convert_general_state_dict({legacy: _tensor(1)})
    assert list(converted) == [expected]


def test_the_bare_camera_token_is_dropped_not_renamed():
    """The old ``camera_token`` was superseded by ``camera_token_extra``. If
    it were merely renamed, the two would collide and the wrong one would
    win, silently swapping the reference-view token."""
    state = {
        "model.backbone.pretrained.camera_token": _tensor(1),
        "model.backbone.pretrained.camera_token_extra": _tensor(2),
    }
    converted = convert_general_state_dict(state)
    assert list(converted) == ["model.backbone.pretrained.camera_token"]
    torch.testing.assert_close(converted["model.backbone.pretrained.camera_token"], _tensor(2))


def test_tensors_are_carried_across_untouched():
    state = {"module.a.net.fc_rot.weight": _tensor(7)}
    (value,) = convert_general_state_dict(state).values()
    assert value is state["module.a.net.fc_rot.weight"]


def test_unrelated_keys_are_left_alone():
    state = {
        "model.head.scratch.layer1.weight": _tensor(1),
        "model.backbone.norm.bias": _tensor(2),
    }
    assert set(convert_general_state_dict(state)) == set(state)


def test_the_renames_are_substring_replacements_not_prefix_matches():
    """``str.replace`` rewrites every occurrence anywhere in the key. Tightening
    this to a prefix match would change which checkpoints load, so the current
    behaviour is recorded deliberately."""
    converted = convert_general_state_dict({"a.module.b.net.c": _tensor(1)})
    assert list(converted) == ["a.model.b.backbone.c"]


def test_several_renames_compose_on_one_key():
    converted = convert_general_state_dict(
        {"module.all_heads.camera_head.more_mlps.0.weight": _tensor(1)}
    )
    assert list(converted) == ["model.cam_dec.backbone.0.weight"]


def test_no_two_renames_collide_on_a_realistic_key_set():
    """Every legacy key must land on a distinct target; a collision would
    silently drop a tensor."""
    state = {legacy: _tensor(i) for i, (legacy, _) in enumerate(RENAMES)}
    converted = convert_general_state_dict(state)
    expected = {expected for _, expected in RENAMES}
    assert set(converted) == expected


def test_the_metric_converter_is_the_general_one_behind_a_module_prefix():
    """Metric checkpoints were saved without the DDP wrapper, so the prefix is
    added back before the shared table runs."""
    state = {"head.weight": _tensor(1), "net.blocks.0.weight": _tensor(2)}
    assert convert_metric_state_dict(state) == convert_general_state_dict(
        {"module." + k: v for k, v in state.items()}
    )
    assert set(convert_metric_state_dict(state)) == {
        "model.head.weight",
        "model.backbone.blocks.0.weight",
    }


# ---------------------------------------------------------------------------
# against a real network
# ---------------------------------------------------------------------------
def _as_legacy(state: dict) -> dict:
    """Rewrite current keys into the released checkpoints' naming.

    Only the renames with an unambiguous inverse are applied. Note the camera
    token: a checkpoint supplies ``camera_token_extra`` and the converter
    drops the plain ``camera_token``, so the model's single ``camera_token``
    parameter is fed by the ``_extra`` entry.
    """
    out = {}
    for key, value in state.items():
        key = key.replace("model.", "module.", 1).replace(".backbone.", ".net.", 1)
        key = key.replace(".pretrained.camera_token", ".pretrained.camera_token_extra")
        out[key] = value
    return out


def test_a_legacy_checkpoint_of_the_real_model_converts_back_exactly(tiny_net):
    """Round trip the real key set through the released naming.

    This is what says the table still covers the architecture: a module
    renamed in the model without a matching entry here shows up as a
    difference in the key sets.
    """
    current = {f"model.{k}": v for k, v in tiny_net.state_dict().items()}
    legacy = _as_legacy(current)
    assert legacy != current

    restored = convert_general_state_dict(legacy)
    assert set(restored) == set(current)


def test_the_converted_keys_load_into_the_real_model(tiny_net):
    """The end of the line: a converted checkpoint has to satisfy
    ``load_state_dict`` with nothing missing and nothing left over."""
    current = {f"model.{k}": v for k, v in tiny_net.state_dict().items()}
    converted = convert_general_state_dict(_as_legacy(current))

    wrapper = nn.Module()
    wrapper.model = tiny_net
    missed, unexpected = wrapper.load_state_dict(converted, strict=False)

    assert not missed and not unexpected


def test_the_camera_token_really_is_a_parameter_of_the_model(tiny_net):
    """Guard the guard above: if the backbone stopped carrying a
    ``camera_token``, the round trip would become vacuous."""
    assert any(k.endswith("pretrained.camera_token") for k in tiny_net.state_dict())


# ---------------------------------------------------------------------------
# load_pretrained_weights
# ---------------------------------------------------------------------------
class _Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.head = nn.Linear(2, 2, bias=False)


def test_weights_are_loaded_and_mismatches_reported(tmp_path):
    target = _Stub()
    weights = torch.full((2, 2), 3.0)
    path = tmp_path / "legacy.pt"
    torch.save({"module.head.weight": weights, "module.head.unknown": torch.zeros(1)}, path)

    missed, unexpected = load_pretrained_weights(target, str(path))

    torch.testing.assert_close(target.model.head.weight.data, weights)
    assert list(missed) == []
    assert list(unexpected) == ["model.head.unknown"]


def test_missing_keys_are_reported_rather_than_raised(tmp_path):
    """The loader is deliberately non-strict, which is exactly why the missed
    list has to be surfaced: silence here means random weights."""
    target = _Stub()
    path = tmp_path / "empty.pt"
    torch.save({}, path)

    missed, unexpected = load_pretrained_weights(target, str(path))

    assert list(missed) == ["model.head.weight"]
    assert list(unexpected) == []


def test_the_metric_flag_selects_the_other_converter(tmp_path):
    target = _Stub()
    weights = torch.full((2, 2), 5.0)
    path = tmp_path / "metric.pt"
    torch.save({"head.weight": weights}, path)  # no module. prefix

    missed, _ = load_pretrained_weights(target, str(path), is_metric=True)

    assert not missed
    torch.testing.assert_close(target.model.head.weight.data, weights)
