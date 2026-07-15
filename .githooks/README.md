# Git hooks

Tracked hooks for this repo. They are **not active until you opt in**:
`core.hooksPath` is local git config and does not travel with a clone.

Enable once per clone:

```sh
git config core.hooksPath .githooks
```

## `pre-commit`

Runs `black --check src/ tests/`, the same gate CI enforces, and rejects the
commit if any file is not black-clean. This keeps formatting drift from
reaching CI.

Fix a rejection with:

```sh
black src/ tests/
git add -u
```

Bypass for a single commit with `git commit --no-verify` (discouraged).
