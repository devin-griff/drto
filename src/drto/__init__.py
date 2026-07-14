# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""drto: dynamic real-time optimization for Pyomo models.

Receding-horizon NMPC and moving horizon estimation for ``pyomo.dae`` models.
Design phase: the framework is recorded in DESIGN.md and AGENTS.md. No
functionality yet.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("drto")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0"

__all__ = ["__version__"]
