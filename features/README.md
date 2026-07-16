# Features

drto is built spec-first: each feature is specified here before it is
implemented. A feature spec is one Markdown file, numbered in order
(`NNN-<name>.md`), with a Status line at the top followed by three sections.

## Status

Lifecycle: ![draft](https://img.shields.io/badge/draft-lightgrey) &rarr; ![ready](https://img.shields.io/badge/ready-blue) &rarr; ![implemented](https://img.shields.io/badge/implemented-yellowgreen) &rarr; ![shipped](https://img.shields.io/badge/shipped-brightgreen)

| Feature | State |
| --- | --- |
| [001 drto.info](001-info.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |
| [002 declarations](002-declarations.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |
| [003 build_objective](003-build_objective.md) | ![ready](https://img.shields.io/badge/ready-blue) |
| [004 dynamic_to_steady_state](004-dynamic_to_steady_state.md) | ![ready](https://img.shields.io/badge/ready-blue) |
| [005 dynamic_optimization](005-dynamic_optimization.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |
| [006 dynamic_simulation](006-dynamic_simulation.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |
| [007 steady_state_simulation](007-steady_state_simulation.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |
| [008 steady_state_optimization](008-steady_state_optimization.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |
| [009 initialize_steady_state](009-initialize_steady_state.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |
| [010 cold_start_dynamic](010-cold_start_dynamic.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |

## Template

- `# <name>`: the capability or transform name, e.g. `drto.build_objective`.
- `**Status:**` at the top: the feature's current state, one of `draft`,
  `ready`, `implemented`, or `shipped`. Keep the status table above in sync.
- `## Description`: a user story, "As a user of DRTO, I want X, so that Y."
- `## Benefit hypothesis`: the value the feature is expected to deliver and
  why it is worth building.
- `## Acceptance criteria`: the concrete, testable conditions the
  implementation must satisfy. They drive the tests and the definition of
  done, which includes the feature's documentation (a `docs/` page and, where
  it applies, an example notebook).

`001-info.md` is the first example.
