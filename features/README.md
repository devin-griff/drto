# Features

drto is built spec-first: each feature is specified here before it is
implemented. A feature spec is one Markdown file, numbered in order
(`NNN-<name>.md`), with a Status line at the top followed by three sections.

## Template

- `# <name>`: the capability or transform name, e.g. `drto.build_objective`.
- `**Status:**` at the top: the feature's current state (for example
  `ready`, meaning the spec is agreed and ready to implement).
- `## Description`: a user story, "As a user of DRTO, I want X, so that Y."
- `## Benefit hypothesis`: the value the feature is expected to deliver and
  why it is worth building.
- `## Acceptance criteria`: the concrete, testable conditions the
  implementation must satisfy. They drive the tests and the definition of
  done.

`001-info.md` is the first example.
