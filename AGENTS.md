# Agent guide for drto

drto is a Python package: receding-horizon NMPC and moving horizon estimation
for `pyomo.dae` models. This file is the entry point for coding agents. Read
it before working in this repo.

## Status: design phase

There is no implementation yet. The design is settled and recorded:

- **DESIGN.md** is the authoritative design record: the six-mode framework,
  the declaration surface, and every locked decision.
- **README.md** is the user-facing overview of the same.

Treat both as the source of truth. Before touching anything on the
declaration or API surface, read DESIGN.md. A decision logged there as
`USER DECISION <date>` is authoritative: do not silently reverse or
reinterpret it. If new work seems to require changing one, surface it and
ask, do not just diverge.

## Repo conventions

Canonical commands (they mirror CI, so local green means CI green; do not
hand-roll black or pytest flags):

- `make dev` -- editable install with dev extras.
- `make lint` -- `black --check` plus `typos`.
- `make test` -- `pytest` with coverage.
- `make check-imports` -- import drto with only base deps.

If `make` is not on PATH (a bare Windows shell), the targets are thin
wrappers: run `black --check --diff src/ tests/`, `typos`, and
`python -m pytest` directly. `make help` (or reading the Makefile) lists the
exact command behind each target.

This is a single pure-Python package that matches its siblings pyomo-cvp and
pyomo-cp. When adding a file, copy the shape of the nearest sibling rather
than inventing a new one.

- **License:** BSD-3-Clause. Every source file carries the two-line header
  `# Copyright (c) 2026 Devin Griffith` /
  `# SPDX-License-Identifier: BSD-3-Clause`.
- **Layout:** hatchling build, `src/drto/` package.
- **Formatting:** Black, line length 88, skip-string-normalization,
  skip-magic-trailing-comma (Pyomo's own settings). Spell-check with `typos`.
- **Versioning:** Keep a Changelog plus SemVer in `CHANGELOG.md`.
- **Optional dependencies** go through
  `pyomo.common.dependencies.attempt_import` so the package imports cleanly
  when a backend (the pounce solver, pyomo-cvp) is absent. Prefer explicit
  declaration over introspection throughout.
- **Do not defer tech debt:** fix deprecated deps, outdated action versions,
  and floating refs in the same pass you notice them.

## Definition of done

A user-facing change is not done until code plus a pinning `pytest`, a bullet
under `## [Unreleased]` in `CHANGELOG.md`, and the relevant
README/docstring/notebook update all land in the same change. See
CONTRIBUTING.md.

## House style

No em dashes anywhere: code, comments, docs, commit messages, changelog.
Short plain sentences. Comments state present-tense constraints and
rationale, not development history. Design history lives in `dev-notes/` and
`DESIGN.md`, not in code comments.

## Intended module map (aspirational, grows from DESIGN.md)

Not built yet; recorded so the first code lands in the right shape:

- The declaration surface (`declare_state`, `declare_control`, the cost and
  boundary declarations, the estimation declarations) is the public API.
- One receding-horizon loop underlies the six modes (steady-state / dynamic
  by simulation / optimization / estimation); the ideal / real-time /
  advanced-step execution variants are variants of dynamic optimization, not
  separate modes.
- The sensitivity fast update rides on pyomo-pounce; control parameterization
  on pyomo-cvp. Both are dependencies, not vendored.

Grow this into a "how to drive drto" table once there is a runnable loop.
