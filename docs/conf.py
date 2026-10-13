"""Sphinx configuration for the mixed signals documentation."""

import dataclasses
from importlib.metadata import version as distribution_version

import sphinxcontrib.bibtex.plugin
from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year import AuthorYearReferenceStyle

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
    "sphinxcontrib.bibtex",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]

# * Docstrings follow the Google convention.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# * The pydantic model configuration is an implementation detail, not API.
autodoc_default_options = {"exclude-members": "model_config"}

myst_enable_extensions = ["colon_fence", "dollarmath"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
    "pydantic": ("https://pydantic.dev/docs/validation/latest", None),
}


def _round_brackets() -> BracketStyle:
    return BracketStyle(left="(", right=")", sep="; ")


@dataclasses.dataclass
class RoundAuthorYearStyle(AuthorYearReferenceStyle):
    """Author-year citations in round brackets, such as Mardia and Jupp (2000)."""

    bracket_textual: BracketStyle = dataclasses.field(default_factory=_round_brackets)
    bracket_parenthetical: BracketStyle = dataclasses.field(
        default_factory=_round_brackets
    )
    bracket_author: BracketStyle = dataclasses.field(default_factory=_round_brackets)
    bracket_label: BracketStyle = dataclasses.field(default_factory=_round_brackets)
    bracket_year: BracketStyle = dataclasses.field(default_factory=_round_brackets)


sphinxcontrib.bibtex.plugin.register_plugin(
    "sphinxcontrib.bibtex.style.referencing", "author_year_round", RoundAuthorYearStyle
)

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year_round"

html_theme = "furo"
html_title = "mixed signals"
