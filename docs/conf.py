# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Sphinx configuration for the drto documentation."""
from importlib.metadata import PackageNotFoundError, version as _version

project = "drto"
author = "Devin Griffith"
copyright = "2026, Devin Griffith"

try:
    release = _version("drto")
except PackageNotFoundError:
    release = "0.0.0"
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

# Notebook execution is off: the docs tree carries no notebooks, and the docs
# build installs neither the solvers nor the examples extra that the notebooks
# under examples/ need. Flip to "cache" when a notebook lands in this tree and
# the build can solve it.
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]

html_theme = "sphinx_book_theme"
html_title = "drto"
html_baseurl = "https://docs.drto.io/"
html_theme_options = {
    "repository_url": "https://github.com/devin-griff/drto",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
}

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
