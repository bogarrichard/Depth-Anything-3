"""CLI surface in cli.py.

Typer builds each parser by resolving type annotations at runtime, so simply
constructing the parsers exercises every Option declaration. The signature
test below is the regression guard for the `ref_view_strategy` NameError:
all four affected commands forwarded a keyword that neither they nor
`run_inference` defined.
"""

import ast
import inspect
import pathlib
import pytest
from typer.testing import CliRunner

from depth_anything_3.cli import app
from depth_anything_3.services.inference_service import run_inference

COMMANDS = ["auto", "image", "images", "colmap", "video"]
CLI_SOURCE = pathlib.Path(inspect.getfile(__import__("depth_anything_3.cli", fromlist=["cli"])))


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


def test_top_level_help(runner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


@pytest.mark.parametrize("command", COMMANDS)
def test_subcommand_help(runner, command):
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output


def _run_inference_calls():
    """Yield (function_name, [keyword names]) for every run_inference call in cli.py."""
    tree = ast.parse(CLI_SOURCE.read_text())
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_inference"
            ):
                yield fn, [kw.arg for kw in node.keywords if kw.arg]


def test_run_inference_is_called_somewhere():
    """Guard the guard: if the call shape changes, the tests below must be updated."""
    assert list(_run_inference_calls()), "no run_inference calls found in cli.py"


def test_every_forwarded_keyword_exists_on_run_inference():
    accepted = set(inspect.signature(run_inference).parameters)
    for fn, keywords in _run_inference_calls():
        unknown = set(keywords) - accepted
        assert not unknown, f"{fn.name}() passes unknown keyword(s) to run_inference: {unknown}"


def test_every_forwarded_keyword_is_bound_in_its_command():
    """Each keyword's value must be a name the command actually defines.

    `reference_view_strategy=reference_view_strategy` passed this file's
    syntax check but was an undefined name at runtime.
    """
    tree = ast.parse(CLI_SOURCE.read_text())
    module_names = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    for fn, _ in _run_inference_calls():
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        local = {
            n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        for node in ast.walk(fn):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_inference"
            ):
                continue
            for kw in node.keywords:
                if kw.arg and isinstance(kw.value, ast.Name):
                    assert kw.value.id in params | local | module_names, (
                        f"{fn.name}() forwards undefined name {kw.value.id!r} as {kw.arg!r}"
                    )
