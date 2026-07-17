# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""drto: dynamic real-time optimization for Pyomo models.

Receding-horizon NMPC and moving horizon estimation for ``pyomo.dae`` models.
The design is recorded in DESIGN.md and the feature specs under ``features/``;
the surface fills in feature by feature, starting with the registry
(``drto.info``).
"""
from importlib.metadata import PackageNotFoundError, version

from drto.info import Info, info

try:
    __version__ = version("drto")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0"

__all__ = ["Info", "info", "__version__"]
