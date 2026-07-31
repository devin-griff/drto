# Contributing to drto

This file is about getting a change *merge-ready*. The agent-facing
conventions and the repo map live in [AGENTS.md](AGENTS.md).

## Enable the git hooks (one-time)

```sh
git config core.hooksPath .githooks
```

The `pre-commit` hook runs `black --check`, mirroring CI so formatting drift
never reaches `main`.

## Definition of done for a user-facing change

A change that adds or changes user-visible behavior is not done until **all
three** of these land in the same PR:

1. **Code + test.** The behavior, with a `pytest` that pins it. A test pins
   behavior only if it fails when the behavior is broken or removed; show
   that, not just the green run. Every fallback the change promises
   ("without X", "when Y is absent") gets its own test, because
   optional-dependency paths are the ones no development environment hits
   naturally. When the change creates components another part of the
   package must later find, it records the pairing in the registry; no
   consumer reconstructs another module's component names (gh #27).
2. **CHANGELOG entry.** A bullet under the `## [Unreleased]` section of
   `CHANGELOG.md`, in the user's terms. At release time the section is renamed
   to the version and dated, and every feature sitting at `implemented` whose
   work the release carries flips to `shipped` in the same commit.
3. **Docs.** The docstring, the relevant `docs/` guide or API page, and an
   example notebook where it applies, so the feature is documented where a
   user looks.

When the change touches a real model, run the example and read its numbers,
not just its exit code: the report, the solver log's iteration count, the
absence of warnings, stated in the PR. That is the check CI cannot do.

## Run the CI guards locally before pushing

These mirror the CI lint, test, and docs jobs; run them for fast feedback:

```sh
black --check --diff src/ tests/                          # formatting gate
typos                                                     # spell-check
python -m pytest -q --cov=drto --cov-report=term-missing  # tests
python -m sphinx -b html -W --keep-going docs docs/_build/html  # docs build
```

## House style

No em dashes anywhere: code, comments, docs, commits, changelog. Short plain
sentences. Comments state present-tense rationale, not history; design history
lives in `dev-notes/` and `DESIGN.md`.
