# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""drto: dynamic real-time optimization for Pyomo models.

Receding-horizon NMPC and moving horizon estimation for ``pyomo.dae`` models.
The design is recorded in DESIGN.md and the feature specs under ``features/``;
the surface fills in feature by feature, starting with the registry
(``drto.info``).
"""
from importlib.metadata import PackageNotFoundError, version

from drto.declarations import (
    arrival_cost,
    control,
    disturbance,
    dynamics,
    economic_stage_cost,
    estimated_parameter,
    estimation_stage_cost,
    estimation_terminal_cost,
    horizon,
    initial_condition,
    measurement,
    state,
    steady_state,
    steady_state_control,
    terminal_constraint,
    tracking_stage_cost,
    tracking_terminal_cost,
)
from drto.info import Info, info
from drto.initialize_steady_state import SteadyStateInitReport, initialize_steady_state
from drto.objective import build_objective

# importing registers the drto.* transformations
from drto import dynamic_optimization as _dynamic_optimization  # noqa: F401
from drto import dynamic_to_steady_state as _dynamic_to_steady_state  # noqa: F401
from drto import infinite_horizon as _infinite_horizon  # noqa: F401
from drto import parameterize as _parameterize  # noqa: F401
from drto import steady_state_simulation as _steady_state_simulation  # noqa: F401

try:
    __version__ = version("drto")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0"

__all__ = [
    "Info",
    "info",
    "build_objective",
    "initialize_steady_state",
    "SteadyStateInitReport",
    "horizon",
    "state",
    "dynamics",
    "control",
    "tracking_stage_cost",
    "economic_stage_cost",
    "tracking_terminal_cost",
    "initial_condition",
    "terminal_constraint",
    "steady_state",
    "steady_state_control",
    "estimated_parameter",
    "disturbance",
    "measurement",
    "estimation_stage_cost",
    "estimation_terminal_cost",
    "arrival_cost",
    "__version__",
]
