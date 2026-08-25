# Agent guide for drto

drto is a Python package: receding-horizon NMPC and moving horizon estimation
for `pyomo.dae` models. This file is the entry point for coding agents. Read
it before working in this repo.

## Status: alpha, implementation in progress

The surface lands one feature at a time. `features/README.md` carries the
per-feature status and is the only place that tracks it: do not restate
implementation status here or in README.md, since a second copy goes stale
silently. The design is settled and recorded:

- **features/** is the authoritative design record, one numbered spec
  per feature, each stating the surface and its acceptance criteria.
- **README.md** is the user-facing overview, including the six-mode
  framework.

Treat them as the source of truth. Before touching anything on the
declaration or API surface, read the feature's spec. A spec states the
design as it stands now, so if new work changes the design, amend the
spec to match rather than diverging from it silently.

Development is spec-first: each feature is specified in `features/` before it
is implemented. Read the feature's spec and build to its acceptance criteria,
which drive the tests and the definition of done. See `features/README.md`.

## Repo conventions

Canonical commands (they mirror CI, so local green means CI green; do not
hand-roll black or pytest flags):

- `python -m pip install -e ".[dev,docs]"` -- editable install with dev and docs extras.
- `black --check --diff src/ tests/` then `typos` -- lint.
- `python -m pytest -q --cov=drto --cov-report=term-missing` -- test with coverage.
- `python -m sphinx -b html -W --keep-going docs docs/_build/html` -- docs build, warnings as errors.
- `python -c "import drto; print('drto', drto.__version__)"` -- import drto with only base deps.

These mirror the CI lint, test, docs, and import-base jobs. CI also
runs a min-deps job at the floor (Python 3.10, `pyomo==6.10.1`), and
`.github/workflows/ci.yml` is the source of truth for the exact steps.

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
under `## [Unreleased]` in `CHANGELOG.md`, and its documentation (docstrings, a
`docs/` guide or API page, and an example notebook where it applies) all land in
the same change. See CONTRIBUTING.md.

A code review or a multi-session task is not done until its `dev-notes/`
tracker records every item with a verification receipt (see
`dev-notes/README.md`).

## House style

No em dashes anywhere: code, comments, docs, commit messages, changelog.
Short plain sentences. Comments state present-tense constraints and
rationale, not development history. Design history lives in `dev-notes/`, not
in code comments.

## Module map

- The declaration surface (bare nouns: `horizon`, `state`, `dynamics`,
  `control`, the cost and boundary declarations, the paired steady-state
  targets, and the estimation declarations) is the public API. Each
  function serves tagging, wrapping, and (constraint roles) the decorator
  form; see feature 002.
- One receding-horizon loop underlies the six modes (steady-state / dynamic
  by simulation / optimization / estimation); the ideal / nonideal /
  advanced-step execution variants are variants of dynamic optimization, not
  separate modes.
- The sensitivity fast update is built on pyomo-pounce, and control
  parameterization on pyomo-cvp. Both are dependencies, not vendored.
