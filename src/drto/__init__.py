# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""drto: dynamic real-time optimization for Pyomo models.

Receding-horizon NMPC and moving horizon estimation for ``pyomo.dae`` models.
The design is recorded in DESIGN.md and the feature specs under ``features/``;
the surface fills in feature by feature, starting with the registry
(``drto.info``).
"""
from importlib.metadata import PackageNotFoundError, version

from drto.declarations import dynamics, control, economic_stage_cost, horizon, initial_condition, state, steady_state, steady_state_control, terminal_constraint, tracking_stage_cost, tracking_terminal_cost
from drto.info import Info, info
from drto.objective import build_objective

# importing registers the drto.* transformations
from drto import infinite_horizon as _infinite_horizon  # noqa: F401
from drto import parameterize as _parameterize  # noqa: F401

try:
    __version__ = version("drto")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0"

__all__ = ["Info", "info", "build_objective", "horizon", "state", "dynamics", "control", "tracking_stage_cost", "economic_stage_cost", "tracking_terminal_cost", "initial_condition", "terminal_constraint", "steady_state", "steady_state_control", "__version__"]
