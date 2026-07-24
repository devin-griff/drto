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

1. **Code + test.** The behavior, with a `pytest` that pins it.
2. **CHANGELOG entry.** A bullet under the `## [Unreleased]` section of
   `CHANGELOG.md`, in the user's terms. At release time the section is renamed
   to the version and dated, and every feature sitting at `implemented` whose
   work the release carries flips to `shipped` in the same commit.
3. **Docs.** The docstring, the relevant `docs/` guide or API page, and an
   example notebook where it applies, so the feature is documented where a
   user looks.

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
