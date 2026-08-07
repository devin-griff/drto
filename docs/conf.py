# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Sphinx configuration for the drto documentation."""
import json
import shutil
from importlib.metadata import PackageNotFoundError, version as _version
from pathlib import Path

# The example notebooks render into the docs from their committed outputs
# (execution stays off below). They live under examples/ at the repo root,
# and Sphinx only reads its own source tree, so the curated set is copied
# into docs/notebooks/ (gitignored) at build time. Keep this list in sync
# with the toctree in examples.md.
_NOTEBOOKS = [
    "first_order",
    "hicks",
    "hicks_inf",
    "hicks_dynamic_optimization",
    "hicks_dynamic_simulation",
    "hicks_steady_state",
    "hicks_initialize",
    "quad_tank",
    "cart_pole",
    "double_column",
    "idaes_cstr",
    "cstr_cold_start",
    "cstr_warm_start",
    "cstr_advanced_step",
    "cstr_ideal_nmpc",
    "check_index",
]
_here = Path(__file__).parent
_nb_dst = _here / "notebooks"
(_nb_dst / "models").mkdir(parents=True, exist_ok=True)


def _merge_stream_outputs(path):
    """Join a cell's consecutive stdout entries into one output.

    Execution flushes a solve's output in chunks and nbformat saves each
    chunk as its own stream output; the renderer draws every output as a
    separate block, so a long solver log arrives as a column of one-line
    boxes. Joining them here keeps the committed notebooks exactly as
    they were executed and leaves the docs with one block per log, which
    the stylesheet then gives a fixed height and a scrollbar.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        merged = []
        for out in cell.get("outputs", []):
            if (
                merged
                and out.get("output_type") == "stream"
                and merged[-1].get("output_type") == "stream"
                and out.get("name") == merged[-1].get("name")
            ):
                merged[-1]["text"] = list(merged[-1]["text"]) + list(out["text"])
            else:
                merged.append(out)
        if "outputs" in cell:
            cell["outputs"] = merged
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


for _name in _NOTEBOOKS:
    shutil.copy2(_here.parent / "examples" / f"{_name}.ipynb", _nb_dst)
    _merge_stream_outputs(_nb_dst / f"{_name}.ipynb")
# the notebooks link their model modules; ship them as downloads
for _py in (_here.parent / "examples" / "models").glob("*.py"):
    if _py.stem != "asu":  # work in progress, not in any shipped notebook
        shutil.copy2(_py, _nb_dst / "models")

exclude_patterns = ["_build"]

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

# Notebook execution is off: the notebooks are committed executed, so the
# docs render their real outputs, and the docs build installs neither the
# solvers nor IDAES.
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["drto-registry.css", "theme-toggle.css"]
html_js_files = ["theme-toggle.js"]
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
