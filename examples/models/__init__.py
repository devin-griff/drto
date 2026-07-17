# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Canonical declared models for the example notebooks.

Each module holds one model as a builder function returning a fully declared
(feature 002) Pyomo model, so notebooks import a model and show
transformations and results rather than construction.
"""
from models.first_order import first_order
from models.hicks import hicks

__all__ = ["first_order", "hicks"]
