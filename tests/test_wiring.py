"""Signature consistency across the call chain.

The ``ref_view_strategy`` incident: four CLI commands forwarded a keyword
that neither they nor ``run_inference`` defined, which made every one of them
fail at runtime while ``--help`` still worked. That class of bug -- a keyword
that exists on one side of a boundary and not the other -- is invisible to
type checkers here (everything is dynamic ``**kwargs`` and typer options) and
invisible to any test that only checks the top and bottom of the stack.

So each hand-off is checked explicitly:

    cli command  ->  run_inference  ->  InferenceService.run_*_inference
                                    ->  DepthAnything3.inference
                                    ->  DepthAnything3.forward
                                    ->  DepthAnything3Net.forward

Everything here is static: it reads source, not behaviour, so it needs no
weights and catches the mismatch before anything is loaded.
"""

import ast
import inspect
import pathlib
import textwrap
import pytest

from depth_anything_3 import cli as cli_module
from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.da3 import DepthAnything3Net, NestedDepthAnything3Net
from depth_anything_3.services.inference_service import InferenceService, run_inference

CLI_SOURCE = pathlib.Path(inspect.getfile(cli_module)).read_text()
CLI_TREE = ast.parse(CLI_SOURCE)

# `run_inference` decides *where* the work happens; those three never travel on.
ROUTING_PARAMS = {"model_dir", "device", "backend_url"}


def _tree_of(obj):
    """Parse one function or method, decorators and indentation included."""
    return ast.parse(textwrap.dedent(inspect.getsource(obj)))


def _functions(tree):
    return [n for n in tree.body if isinstance(n, ast.FunctionDef)]


def _calls_to(scope, name):
    for node in ast.walk(scope):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
            yield node


def _method_calls_to(scope, attr):
    for node in ast.walk(scope):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == attr:
                yield node


def _params(func) -> set[str]:
    return set(inspect.signature(func).parameters)


# ---------------------------------------------------------------------------
# cli command -> run_inference
# ---------------------------------------------------------------------------
def _cli_run_inference_calls():
    """Yield (command_ast, call_ast) for every run_inference call in cli.py."""
    for fn in _functions(CLI_TREE):
        for call in _calls_to(fn, "run_inference"):
            yield fn, call


def test_the_cli_actually_calls_run_inference():
    """Guard the guard: if the call shape changes, everything below is vacuous."""
    calls = list(_cli_run_inference_calls())
    assert calls, "no run_inference calls found in cli.py"
    assert {fn.name for fn, _ in calls} == {"auto", "image", "images", "colmap", "video"}


def test_every_forwarded_keyword_exists_on_run_inference():
    accepted = _params(run_inference)
    for fn, call in _cli_run_inference_calls():
        unknown = {kw.arg for kw in call.keywords if kw.arg} - accepted
        assert not unknown, f"{fn.name}() passes unknown keyword(s) to run_inference: {unknown}"


def test_every_forwarded_keyword_is_bound_in_its_command():
    """Each keyword's value must be a name the command actually defines.

    ``reference_view_strategy=reference_view_strategy`` passed a syntax check
    but was an undefined name at runtime.
    """
    module_names = {
        n.id
        for n in ast.walk(CLI_TREE)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    for fn, call in _cli_run_inference_calls():
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        local = {
            n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        for kw in call.keywords:
            if kw.arg and isinstance(kw.value, ast.Name):
                assert kw.value.id in params | local | module_names, (
                    f"{fn.name}() forwards undefined name {kw.value.id!r} as {kw.arg!r}"
                )


def test_every_command_supplies_the_arguments_run_inference_has_no_default_for():
    required = {
        name
        for name, p in inspect.signature(run_inference).parameters.items()
        if p.default is p.empty
    }
    for fn, call in _cli_run_inference_calls():
        passed = {kw.arg for kw in call.keywords if kw.arg}
        assert required <= passed, f"{fn.name}() omits {sorted(required - passed)}"


def test_every_command_routes_the_backend_flag():
    """``--use-backend`` is what turns ``backend_url`` from a default string
    into ``None``. A command that forwarded the raw option would silently
    always talk to the backend."""
    for fn, call in _cli_run_inference_calls():
        forwarded = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        assert "backend_url" in forwarded, fn.name
        value = forwarded["backend_url"]
        assert isinstance(value, ast.Name) and value.id == "final_backend_url", (
            f"{fn.name}() forwards {ast.unparse(value)} instead of the gated URL"
        )


# ---------------------------------------------------------------------------
# run_inference -> InferenceService
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method", [InferenceService.run_local_inference, InferenceService.run_backend_inference]
)
def test_run_inference_forwards_everything_it_was_given(method):
    """Both service methods must receive every non-routing parameter. An
    option that stops here is an option the CLI advertises and ignores."""
    source = _tree_of(run_inference)
    calls = list(_method_calls_to(source, method.__name__))
    assert len(calls) == 1, f"expected exactly one {method.__name__} call"

    forwarded = {kw.arg for kw in calls[0].keywords if kw.arg}
    accepted = _params(method) - {"self"}
    expected = _params(run_inference) - ROUTING_PARAMS

    assert forwarded - accepted == set(), (
        f"{method.__name__} does not accept {forwarded - accepted}"
    )
    assert expected - forwarded == set(), (
        f"{method.__name__} never receives {expected - forwarded}"
    )


def test_the_two_service_methods_share_one_parameter_set():
    """They are alternatives chosen by a single flag; a caller must be able to
    switch between them without knowing which options survive."""
    local = _params(InferenceService.run_local_inference) - {"self"}
    backend = _params(InferenceService.run_backend_inference) - {"self", "backend_url"}
    assert local == backend


# ---------------------------------------------------------------------------
# InferenceService -> DepthAnything3.inference
# ---------------------------------------------------------------------------
def _dict_literal_keys(scope, variable):
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if variable in targets:
                return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError(f"no dict literal assigned to {variable!r}")


def _subscript_assigned_keys(scope, variable):
    keys = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == variable
                    and isinstance(target.slice, ast.Constant)
                ):
                    keys.add(target.slice.value)
    return keys


def test_local_inference_only_passes_keys_the_api_accepts():
    source = _tree_of(InferenceService.run_local_inference)
    keys = _dict_literal_keys(source, "inference_kwargs") | _subscript_assigned_keys(
        source, "inference_kwargs"
    )
    accepted = _params(DepthAnything3.inference) - {"self"}
    assert keys, "no inference_kwargs found"
    assert keys <= accepted, f"unknown keyword(s) for DepthAnything3.inference: {keys - accepted}"


def test_local_inference_passes_on_every_option_it_was_given():
    source = _tree_of(InferenceService.run_local_inference)
    keys = _dict_literal_keys(source, "inference_kwargs") | _subscript_assigned_keys(
        source, "inference_kwargs"
    )
    own = _params(InferenceService.run_local_inference) - {"self", "image_paths"}
    assert own <= keys, f"run_local_inference swallows {sorted(own - keys)}"


# ---------------------------------------------------------------------------
# InferenceService -> the HTTP backend
# ---------------------------------------------------------------------------
def _backend_payload_keys():
    source = _tree_of(InferenceService.run_backend_inference)
    return _dict_literal_keys(source, "payload") | _subscript_assigned_keys(source, "payload")


def test_the_backend_payload_is_not_empty():
    assert _backend_payload_keys()


def test_every_payload_key_is_a_field_the_backend_model_accepts():
    """A key the server model does not declare is dropped without a word, so
    the option appears to work and does nothing.

    ``use_ray_pose`` and ``ref_view_strategy`` were exactly that: sent by the
    client, undeclared on the server, silently discarded by pydantic.
    """
    from depth_anything_3.services.backend import InferenceRequest

    accepted = set(InferenceRequest.model_fields)
    assert _backend_payload_keys() <= accepted


def test_every_declared_request_field_reaches_the_model():
    """The other direction: a field the request declares but the task builder
    never reads is an option the server advertises and ignores."""
    from depth_anything_3.services import backend as backend_module

    source = _tree_of(backend_module._run_inference_task)
    forwarded = _dict_literal_keys(source, "inference_kwargs") | _subscript_assigned_keys(
        source, "inference_kwargs"
    )
    # `image_paths` is renamed to `image` on the way in; `export_dir` is
    # validated before being added under its own name.
    declared = set(backend_module.InferenceRequest.model_fields) - {"image_paths"}
    assert declared <= forwarded, f"never reaches the model: {sorted(declared - forwarded)}"


def test_the_backend_rejects_export_formats_the_dispatcher_cannot_handle():
    """The request model is the server-trusted enum for ``export_format``;
    without it a client string reaches the dispatcher unchecked."""
    import pydantic

    from depth_anything_3.services.backend import InferenceRequest
    from depth_anything_3.utils.export import SUPPORTED_EXPORT_FORMATS

    InferenceRequest(image_paths=["a.png"], export_format="mini_npz-glb")
    with pytest.raises(pydantic.ValidationError):
        InferenceRequest(image_paths=["a.png"], export_format="rm -rf")
    with pytest.raises(pydantic.ValidationError):
        InferenceRequest(image_paths=["a.png"], export_format="glb-not_a_format")
    for fmt in SUPPORTED_EXPORT_FORMATS:
        InferenceRequest(image_paths=["a.png"], export_format=fmt)


# ---------------------------------------------------------------------------
# DepthAnything3 -> DepthAnything3Net
# ---------------------------------------------------------------------------
def test_the_api_calls_the_network_in_the_networks_own_argument_order():
    """``DepthAnything3.forward`` passes seven arguments *positionally*.
    Reordering the network's signature would keep every shape valid and
    silently swap, say, ``infer_gs`` with ``use_ray_pose``."""
    source = _tree_of(DepthAnything3.forward)
    calls = list(_method_calls_to(source, "model"))
    assert len(calls) == 1, "expected exactly one self.model(...) call"

    positional = [ast.unparse(a) for a in calls[0].args]
    net_params = [
        p for p in inspect.signature(DepthAnything3Net.forward).parameters if p != "self"
    ]
    assert not calls[0].keywords, "the call is positional; the check below assumes that"
    # Slot 0 is the image tensor: `image` on the API, `x` on the network.
    assert positional[0] == "image" and net_params[0] == "x"
    assert positional[1:] == net_params[1 : len(positional)], (
        f"positional call {positional} does not line up with {net_params}"
    )


def test_the_nested_network_is_a_drop_in_replacement():
    """``create_object`` may hand back either class; ``DepthAnything3`` calls
    them identically, so their forward signatures have to agree."""
    plain = inspect.signature(DepthAnything3Net.forward)
    nested = inspect.signature(NestedDepthAnything3Net.forward)
    assert list(plain.parameters) == list(nested.parameters)
    for name in plain.parameters:
        assert plain.parameters[name].default == nested.parameters[name].default, name


def test_the_public_api_exposes_the_networks_switches():
    """Anything the network branches on must be reachable from
    ``DepthAnything3.inference``; otherwise the branch is dead in practice."""
    net_switches = _params(DepthAnything3Net.forward) - {"self", "x"}
    api_params = _params(DepthAnything3.inference)
    missing = net_switches - api_params - {"extrinsics", "intrinsics"}
    assert not missing, f"unreachable from the public API: {sorted(missing)}"
