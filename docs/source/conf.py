# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "Depth Anything 3"
copyright = "2025 ByteDance Ltd. and/or its affiliates"
author = "Depth Anything 3 contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinxcontrib.typer",
    "sphinx_autodoc_typehints",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

source_suffix = {
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# --- autodoc -----------------------------------------------------------
# The COLMAP export path needs `pycolmap`, an optional extra with no
# manylinux wheel matching every platform -- kept out of the docs build
# entirely rather than mocked, so the reference only documents what a plain
# `pip install depth-anything-3` actually gives you.
autodoc_mock_imports = ["pycolmap"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# --- HTML ----------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
html_title = "Depth Anything 3"
