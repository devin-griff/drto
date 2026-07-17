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
    declare_continuous_dynamics,
    declare_control,
    declare_economic_stage_cost,
    declare_initial_condition,
    declare_state,
    declare_steady_state,
    declare_steady_state_control,
    declare_terminal_constraint,
    declare_time,
    declare_tracking_stage_cost,
    declare_tracking_terminal_cost,
)
from drto.info import Info, info

try:
    __version__ = version("drto")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0"

__all__ = [
    "Info",
    "info",
    "declare_time",
    "declare_state",
    "declare_continuous_dynamics",
    "declare_control",
    "declare_tracking_stage_cost",
    "declare_economic_stage_cost",
    "declare_tracking_terminal_cost",
    "declare_initial_condition",
    "declare_terminal_constraint",
    "declare_steady_state",
    "declare_steady_state_control",
    "__version__",
]
