"""The config mechanism: ``cfg.py`` and the shipped presets in ``configs/``.

Every model in this package is built by ``create_object`` walking an
OmegaConf tree and calling whatever ``__object__`` names. That indirection
means a renamed class, a dropped constructor argument or a mistyped
dimension shows up only when someone downloads gigabytes of weights and runs
the giant preset -- unless something checks the presets statically.

The heavy lifting here is :func:`test_every_preset_can_actually_be_built`,
which resolves each shipped preset, imports every class it names and checks
that the arguments the config passes are ones the constructor accepts. It
costs no weights and no GPU, and it fails on exactly the refactor that would
otherwise be discovered in production.
"""

import inspect
from pathlib import Path
import pytest
from omegaconf import DictConfig, OmegaConf

from depth_anything_3 import registry as registry_module
from depth_anything_3.cfg import (
    create_object,
    import_item,
    load_config,
    to_dict_recursive,
)
from depth_anything_3.registry import MODEL_REGISTRY

PRESETS = sorted(MODEL_REGISTRY)
PACKAGE_ROOT = Path(registry_module.__file__).resolve().parent


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------
def test_registry_lists_every_shipped_yaml_and_nothing_else():
    configs = {p.stem: p for p in (PACKAGE_ROOT / "configs").glob("*.yaml")}
    assert set(MODEL_REGISTRY) == set(configs)
    for name, path in MODEL_REGISTRY.items():
        assert Path(path).is_file(), name


def test_registry_is_sorted():
    """``get_all_models`` promises a sorted OrderedDict; the gradio dropdown
    and the CLI help both render it in that order."""
    assert list(MODEL_REGISTRY) == sorted(MODEL_REGISTRY)


def test_the_evaluation_default_strategy_is_one_the_model_accepts():
    """``constants.EVAL_REF_VIEW_STRATEGY`` is fed straight to the backbone by
    ``bench/evaluator.py``. The comment beside it still offers "auto" and
    "mid", neither of which the selector knows -- so the value itself has to
    be checked against the real enum."""
    import typing

    from depth_anything_3.model.reference_view_selector import RefViewStrategy
    from depth_anything_3.utils.constants import EVAL_REF_VIEW_STRATEGY

    assert EVAL_REF_VIEW_STRATEGY in typing.get_args(RefViewStrategy)


def test_the_benchmark_config_default_strategy_is_valid():
    import typing

    from depth_anything_3.model.reference_view_selector import RefViewStrategy

    cfg = OmegaConf.load(PACKAGE_ROOT / "bench" / "configs" / "eval_bench.yaml")
    assert cfg.eval.ref_view_strategy in typing.get_args(RefViewStrategy)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------
def test_module_path_and_file_path_load_the_same_config():
    by_module = load_config("depth_anything_3.configs.da3-small")
    by_file = load_config(MODEL_REGISTRY["da3-small"])
    assert to_dict_recursive(by_module) == to_dict_recursive(by_file)


def test_dotlist_overrides_are_applied():
    cfg = load_config("depth_anything_3.configs.da3-small", argv=["net.name=vitl"])
    assert cfg.net.name == "vitl"


def test_inheritance_merges_parents_and_lets_the_child_win(tmp_path):
    parent = tmp_path / "parent.yaml"
    parent.write_text("a: 1\nb: 2\nnested:\n  x: 10\n  y: 20\n")
    child = tmp_path / "child.yaml"
    child.write_text(f"__inherit__: {parent}\nb: 99\nnested:\n  y: 42\n")

    cfg = load_config(str(child))

    assert cfg.a == 1  # inherited
    assert cfg.b == 99  # overridden
    assert cfg.nested.x == 10 and cfg.nested.y == 42  # merged, not replaced
    assert "__inherit__" not in cfg


def test_inheritance_from_several_parents_applies_them_left_to_right(tmp_path):
    first = tmp_path / "first.yaml"
    first.write_text("a: 1\nshared: from_first\n")
    second = tmp_path / "second.yaml"
    second.write_text("b: 2\nshared: from_second\n")
    child = tmp_path / "child.yaml"
    child.write_text(f"__inherit__:\n  - {first}\n  - {second}\n")

    cfg = load_config(str(child))

    assert cfg.a == 1 and cfg.b == 2
    assert cfg.shared == "from_second"


def test_inheritance_is_resolved_at_every_depth(tmp_path):
    """``da3nested-giant-large`` inherits inside a nested key, so the
    resolver has to recurse rather than only look at the root."""
    parent = tmp_path / "parent.yaml"
    parent.write_text("value: 7\n")
    child = tmp_path / "child.yaml"
    child.write_text(f"branch:\n  __inherit__: {parent}\n")

    cfg = load_config(str(child))

    assert cfg.branch.value == 7


def test_the_eval_resolver_is_registered():
    """``cfg.py`` registers an ``eval`` resolver at import time and swallows
    the error if it fails. Configs are free to use it, so check it works."""
    cfg = OmegaConf.create({"n": "${eval:2 * 3 + 1}"})
    assert cfg.n == 7


# ---------------------------------------------------------------------------
# create_object
# ---------------------------------------------------------------------------
class _AsParams:
    def __init__(self, alpha, beta=2):
        self.alpha, self.beta = alpha, beta


class _AsConfig:
    def __init__(self, cfg):
        self.cfg = cfg


def _object_cfg(name: str, args: str | None = None, **extra) -> DictConfig:
    spec = {"path": __name__, "name": name}
    if args is not None:
        spec["args"] = args
    return OmegaConf.create({"__object__": spec, **extra})


def test_as_params_passes_the_sibling_keys_as_keyword_arguments():
    obj = create_object(_object_cfg("_AsParams", "as_params", alpha=5))
    assert isinstance(obj, _AsParams)
    assert (obj.alpha, obj.beta) == (5, 2)


def test_as_config_is_the_default_and_hands_over_the_whole_node():
    obj = create_object(_object_cfg("_AsConfig", alpha=5))
    assert isinstance(obj, _AsConfig)
    assert obj.cfg.alpha == 5
    assert "__object__" in obj.cfg


def test_as_params_strips_the_object_key_before_calling():
    """It must not leak ``__object__`` into the constructor -- that would hit
    every ``as_params`` class as an unexpected keyword."""
    obj = create_object(_object_cfg("_AsParams", "as_params", alpha=1))
    assert not hasattr(obj, "__object__")


def test_an_unknown_args_mode_is_rejected():
    with pytest.raises(NotImplementedError):
        create_object(_object_cfg("_AsParams", "as_something_else", alpha=1))


def test_import_item_resolves_a_dotted_path():
    assert import_item("depth_anything_3.cfg", "create_object") is create_object


# ---------------------------------------------------------------------------
# the shipped presets
# ---------------------------------------------------------------------------
def _object_nodes(node, prefix=""):
    """Yield (path, node) for every dict in the tree carrying ``__object__``."""
    if isinstance(node, dict):
        if "__object__" in node:
            yield prefix or "<root>", node
        for key, value in node.items():
            yield from _object_nodes(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _object_nodes(value, f"{prefix}[{i}]")


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_resolves_without_leftover_inheritance(preset):
    cfg = to_dict_recursive(load_config(MODEL_REGISTRY[preset]))
    assert "__inherit__" not in str(cfg), f"{preset} still carries an unresolved __inherit__"
    assert list(_object_nodes(cfg)), f"{preset} declares no __object__ to build"


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_can_actually_be_built(preset):
    """Import every class a preset names and check the config's keys against
    its signature -- the constructor call, minus the gigabytes of weights."""
    cfg = to_dict_recursive(load_config(MODEL_REGISTRY[preset]))
    for where, node in _object_nodes(cfg):
        spec = node["__object__"]
        item = import_item(spec["path"], spec["name"])  # raises if renamed or moved
        if spec.get("args", "as_config") != "as_params":
            continue
        signature = inspect.signature(item)
        if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
            continue
        accepted = set(signature.parameters)
        passed = set(node) - {"__object__"}
        assert passed <= accepted, (
            f"{preset}:{where} passes {sorted(passed - accepted)} to "
            f"{spec['name']}, which does not accept it"
        )
        required = {
            name
            for name, p in signature.parameters.items()
            if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        }
        assert required <= passed, (
            f"{preset}:{where} omits required argument(s) "
            f"{sorted(required - passed)} of {spec['name']}"
        )


@pytest.mark.parametrize("preset", PRESETS)
def test_gaussian_head_width_matches_the_adapter(preset):
    """``DepthAnything3Net.__init__`` asserts ``gs_head.output_dim ==
    gs_adapter.d_in + 1``. The adapter is a few buffers, so the arithmetic can
    be checked here without building the 1.4B-parameter backbone it sits on."""
    cfg = load_config(MODEL_REGISTRY[preset])
    if "gs_head" not in cfg or "gs_adapter" not in cfg:
        pytest.skip(f"{preset} has no gaussian branch")
    adapter = create_object(cfg.gs_adapter)
    assert cfg.gs_head.output_dim == adapter.d_in + 1


@pytest.mark.parametrize("preset", PRESETS)
def test_head_input_width_matches_the_backbone_output(preset):
    """``cat_token`` doubles the token width the heads see. Getting the two
    out of step is a silent config error that only surfaces as a shape
    mismatch deep inside the DPT fusion."""
    from depth_anything_3.model.dinov2 import vision_transformer as vit

    cfg = load_config(MODEL_REGISTRY[preset])
    if "net" not in cfg:
        pytest.skip(f"{preset} is a nested preset with no direct backbone")
    factory = {
        "vits": vit.vit_small,
        "vitb": vit.vit_base,
        "vitl": vit.vit_large,
        "vitg": vit.vit_giant2,
    }[cfg.net.name]
    # depth=1 builds a single block: enough to read the real embed_dim without
    # allocating the giant preset's 1.1B parameters.
    embed_dim = factory(patch_size=14, img_size=518, depth=1).embed_dim
    expected = embed_dim * 2 if cfg.net.get("cat_token", False) else embed_dim
    assert cfg.head.dim_in == expected
    if "cam_dec" in cfg:
        assert cfg.cam_dec.dim_in == expected
    if "cam_enc" in cfg:
        assert cfg.cam_enc.dim_out == embed_dim


@pytest.mark.parametrize("preset", PRESETS)
def test_requested_backbone_layers_exist(preset):
    """``out_layers`` indexes the transformer blocks. An index past the last
    block yields fewer feature maps than the DPT head unpacks, which surfaces
    as an IndexError deep in the head instead of a config error."""
    from depth_anything_3.model.dinov2 import vision_transformer as vit

    cfg = load_config(MODEL_REGISTRY[preset])
    if "net" not in cfg:
        pytest.skip(f"{preset} is a nested preset with no direct backbone")
    depth = (
        inspect.signature(
            {
                "vits": vit.vit_small,
                "vitb": vit.vit_base,
                "vitl": vit.vit_large,
                "vitg": vit.vit_giant2,
            }[cfg.net.name]
        )
        .parameters["depth"]
        .default
    )
    assert len(cfg.net.out_layers) == 4, "the DPT heads consume exactly four scales"
    assert all(0 <= layer < depth for layer in cfg.net.out_layers)
    for start in ("alt_start", "qknorm_start", "rope_start"):
        value = cfg.net.get(start, -1)
        assert value == -1 or 0 <= value < depth, (
            f"{preset}: {start}={value} exceeds depth {depth}"
        )
