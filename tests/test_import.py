# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Phase 0 smoke test: the package imports and exposes a version string."""
import drto


def test_version_is_string():
    assert isinstance(drto.__version__, str)
