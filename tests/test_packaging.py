"""Packaging metadata in pyproject.toml.

The `addict` incident: the package imported a module it never declared, so a
clean install failed at `import depth_anything_3.api`. It went unnoticed
because several such modules happened to arrive transitively. This walks
every import in the tree and checks it against the declared set.
"""

import ast
import pathlib
import sys
import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # tomllib is stdlib only from 3.11; tomli is the backport.
    import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())

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

FIRST_PARTY = {
    "depth_anything_3",
    "da3_streaming",
    "loop_utils",
    "fastloop",
    "configs",
    "salad",
    "conftest",
}


def _normalise(name: str) -> str:
    return name.lower().replace("_", "-")


def _declared() -> set[str]:
    project = PYPROJECT["project"]
    specs = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        specs += extra
    names = set()
    for spec in specs:
        base = spec.split(";")[0]
        for sep in (">", "<", "=", "[", "@", "!", "~"):
            base = base.split(sep)[0]
        base = base.strip()
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
    stdlib = set(sys.stdlib_module_names)
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


def test_no_duplicate_dependencies():
    """`uvicorn`, `typer` and `moviepy` were each listed twice once already."""
    deps = PYPROJECT["project"]["dependencies"]
    names = [_normalise(d.split(">")[0].split("<")[0].split("=")[0].strip()) for d in deps]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"duplicated dependencies: {duplicates}"


def test_requires_python_has_an_exclusive_upper_bound():
    """`<=3.13` silently excluded 3.13.1+, because PEP 440 orders 3.13.1 > 3.13."""
    requires = PYPROJECT["project"]["requires-python"]
    assert "<=" not in requires, (
        f"requires-python {requires!r} uses an inclusive upper bound, which "
        "excludes every patch release of the named minor version"
    )


@pytest.mark.parametrize("extra", ["dev", "bench", "app", "gs", "streaming"])
def test_expected_extras_exist(extra):
    assert extra in PYPROJECT["project"]["optional-dependencies"]
