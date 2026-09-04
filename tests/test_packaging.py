"""Packaging metadata in pyproject.toml.

The `addict` incident: the package imported a module it never declared, so a
clean install failed at `import depth_anything_3.api`. It went unnoticed
because several such modules happened to arrive transitively. The scan below
walks every import in the tree and checks it against the declared set --
which is strictly stronger than a hand-rolled sweep, and found two more
(scikit-learn, pyyaml) the manual one had missed.

The rest of this file keeps the metadata honest in the other direction: that
the extras the code tells users to install actually exist, that ``all`` means
all, and that the declared Python range matches what CI proves.
"""

import ast
import pathlib
import re
import sys
import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # tomllib is stdlib only from 3.11; tomli is the backport.
    import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
EXTRAS = PYPROJECT["project"]["optional-dependencies"]

# Distributions whose import name differs from their package name.
IMPORT_NAME = {
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
    "pillow-heif": "pillow_heif",
    "huggingface-hub": "huggingface_hub",
    "faiss-gpu": "faiss",
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "pytest": "pytest",
    "hypothesis": "hypothesis",
}

# Imports that are deliberately not declared, each for a specific reason.
INTENTIONALLY_UNDECLARED = {
    # torch vendors triton on Linux only; declaring it would break other platforms.
    "triton",
    # Not on PyPI: a C++ extension built locally from da3_streaming/fastloop/solve.cpp.
    "sim3solve",
}

# Modules that are stdlib on some supported Python versions but not others.
# The scan below uses the *running* interpreter's stdlib list, so without this
# `tomllib` reads as a third-party import on 3.10 (it entered the stdlib in 3.11).
STDLIB_IN_LATER_PYTHON = {"tomllib"}

FIRST_PARTY = {
    "depth_anything_3",
    "da3_streaming",
    "loop_utils",
    "fastloop",
    "configs",
    "salad",
    "conftest",
    "_oracles",
}


def _normalise(name: str) -> str:
    return name.lower().replace("_", "-")


def _requirement_name(spec: str) -> str:
    base = spec.split(";")[0]
    for sep in (">", "<", "=", "[", "@", "!", "~"):
        base = base.split(sep)[0]
    return base.strip()


def _declared() -> set[str]:
    specs = list(PYPROJECT["project"]["dependencies"])
    for extra in EXTRAS.values():
        specs += extra
    names = set()
    for spec in specs:
        base = _requirement_name(spec)
        if not base:
            continue
        names.add(_normalise(base))
        if base.lower() in IMPORT_NAME:
            names.add(_normalise(IMPORT_NAME[base.lower()]))
    return names


def _source_files():
    for path in ROOT.rglob("*.py"):
        parts = set(path.parts)
        if parts & {".venv", "salad", "build", "dist", "__pycache__"}:
            continue
        yield path


def _third_party_imports():
    """Map top-level third-party import name -> set of files importing it."""
    stdlib = set(sys.stdlib_module_names) | STDLIB_IN_LATER_PYTHON
    found: dict[str, set[pathlib.Path]] = {}
    for path in _source_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative (first-party) import.
                modules = [node.module] if node.module and node.level == 0 else []
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if not top or top in stdlib or top in FIRST_PARTY:
                    continue
                found.setdefault(top, set()).add(path.relative_to(ROOT))
    return found


# ---------------------------------------------------------------------------
# dependencies
# ---------------------------------------------------------------------------
def test_every_third_party_import_is_declared():
    declared = _declared()
    undeclared = {
        name: files
        for name, files in _third_party_imports().items()
        if _normalise(name) not in declared and name not in INTENTIONALLY_UNDECLARED
    }
    assert not undeclared, "undeclared third-party imports: " + ", ".join(
        f"{name} ({sorted(str(f) for f in files)[0]})"
        for name, files in sorted(undeclared.items())
    )


def test_the_scan_finds_something_to_check():
    """Guard the guard: a broken walker would report zero undeclared imports
    and look like a pass."""
    imports = _third_party_imports()
    assert len(imports) > 20
    assert "torch" in imports and "numpy" in imports


def test_every_intentionally_undeclared_import_is_still_imported():
    """The exemption list must not outlive the code that needed it."""
    imported = set(_third_party_imports())
    stale = INTENTIONALLY_UNDECLARED - imported
    assert not stale, f"no longer imported anywhere, drop the exemption: {sorted(stale)}"


def test_no_duplicate_dependencies():
    """`uvicorn`, `typer` and `moviepy` were each listed twice once already."""
    names = [_normalise(_requirement_name(d)) for d in PYPROJECT["project"]["dependencies"]]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"duplicated dependencies: {duplicates}"


def test_no_dependency_is_both_core_and_optional():
    """A package in an extra *and* in the core list makes the extra a lie:
    it is installed either way."""
    core = {_normalise(_requirement_name(d)) for d in PYPROJECT["project"]["dependencies"]}
    for name, specs in EXTRAS.items():
        if name == "all":
            continue
        optional = {_normalise(_requirement_name(s)) for s in specs}
        overlap = (core & optional) - {"depth-anything-3", "pillow"}
        assert not overlap, f"extra {name!r} re-declares core dependencies: {sorted(overlap)}"


# ---------------------------------------------------------------------------
# python version
# ---------------------------------------------------------------------------
def test_requires_python_has_an_exclusive_upper_bound():
    """`<=3.13` silently excluded 3.13.1+, because PEP 440 orders 3.13.1 > 3.13."""
    requires = PYPROJECT["project"]["requires-python"]
    assert "<=" not in requires, (
        f"requires-python {requires!r} uses an inclusive upper bound, which "
        "excludes every patch release of the named minor version"
    )


def test_the_ci_matrix_stays_inside_the_declared_python_range():
    """CI proves the floor and the ceiling; if the declared range moves and
    the matrix does not, the claim stops being tested."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    tested = {m for m in re.findall(r"'(3\.\d+)'", workflow)}
    assert tested, "no python versions found in the CI matrix"

    requires = PYPROJECT["project"]["requires-python"]
    floor = re.search(r">=\s*3\.(\d+)", requires)
    ceiling = re.search(r"<\s*3\.(\d+)", requires)
    assert floor and ceiling, requires
    for version in tested:
        minor = int(version.split(".")[1])
        assert int(floor.group(1)) <= minor < int(ceiling.group(1)), (
            f"CI tests {version}, which requires-python {requires!r} excludes"
        )
    assert f"3.{floor.group(1)}" in tested, "CI does not test the declared floor"


# ---------------------------------------------------------------------------
# extras
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("extra", ["dev", "bench", "app", "gs", "colmap", "streaming"])
def test_expected_extras_exist(extra):
    assert extra in EXTRAS


def test_the_all_extra_bundles_every_optional_extra_except_the_documented_exclusions():
    """``dev`` and ``docs`` are not user-facing extras."""
    referenced = set()
    for spec in EXTRAS["all"]:
        match = re.search(r"\[(.*)\]", spec)
        if match:
            referenced |= {name.strip() for name in match.group(1).split(",")}
    assert referenced == set(EXTRAS) - {"all", "dev", "docs"}


def test_the_all_extra_only_points_at_this_package():
    for spec in EXTRAS["all"]:
        assert _requirement_name(spec) == PYPROJECT["project"]["name"]


def test_every_extra_the_code_tells_users_to_install_exists():
    """Error messages say ``pip install "depth-anything-3[gs]"``. If the extra
    were renamed, that advice would send users to a dead end."""
    pattern = re.compile(r"depth-anything-3\[([a-z,]+)\]")
    mentioned = set()
    for path in _source_files():
        for match in pattern.finditer(path.read_text()):
            mentioned |= {name for name in match.group(1).split(",")}
    assert mentioned, "no install hints found -- the guard needs updating"
    assert mentioned <= set(EXTRAS), (
        f"unknown extras suggested to users: {mentioned - set(EXTRAS)}"
    )


def test_the_dev_extra_can_run_this_suite():
    dev = {_normalise(_requirement_name(s)) for s in EXTRAS["dev"]}
    assert {"pytest", "hypothesis"} <= dev


# ---------------------------------------------------------------------------
# test configuration
# ---------------------------------------------------------------------------
def test_the_suite_can_import_its_own_helpers():
    """``tests/`` on ``pythonpath`` is what makes ``from conftest import ...``
    and ``from _oracles import ...`` work in every file here."""
    config = PYPROJECT["tool"]["pytest"]["ini_options"]
    assert "tests" in config["pythonpath"]
    assert config["testpaths"] == ["tests"]


def test_markers_are_strict():
    """``--strict-markers`` turns a typo in a ``@pytest.mark`` into an error
    instead of a silently ignored decorator."""
    assert "--strict-markers" in PYPROJECT["tool"]["pytest"]["ini_options"]["addopts"]
