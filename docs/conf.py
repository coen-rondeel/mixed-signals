"""Sphinx configuration for the mixed signals documentation."""

from importlib.metadata import version as distribution_version

project = "mixed signals"
author = "coen-rondeel"
copyright = f"2026, {author}"
release = distribution_version("mixed-signals")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]

# * Docstrings follow the Google convention.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"

myst_enable_extensions = ["colon_fence", "dollarmath"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

html_theme = "furo"
html_title = "mixed signals"
