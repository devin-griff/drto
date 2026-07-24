# Features

drto is built spec-first: each feature is specified here before it is
implemented. A feature spec is one Markdown file, numbered in order
(`NNN-<name>.md`), with a Status line at the top followed by three sections.

## Status

Lifecycle: ![draft](https://img.shields.io/badge/draft-lightgrey) &rarr; ![ready](https://img.shields.io/badge/ready-blue) &rarr; ![implemented](https://img.shields.io/badge/implemented-yellowgreen) &rarr; ![shipped](https://img.shields.io/badge/shipped-brightgreen)

`implemented` means the work is merged to main; `shipped` means a tagged
release carries it. The flip to `shipped` happens in the release commit,
alongside the CHANGELOG section rename.

| Feature | State |
| --- | --- |
| [001 drto.info](001-drto.info.md) | ![shipped](https://img.shields.io/badge/shipped-brightgreen) |
| [002 Dynamic optimization and simulation declarations](002-dynamic_optimization_and_simulation_declarations.md) | ![shipped](https://img.shields.io/badge/shipped-brightgreen) |
| [003 drto.build_objective](003-drto.build_objective.md) | ![shipped](https://img.shields.io/badge/shipped-brightgreen) |
| [004 drto.infinite_horizon](004-drto.infinite_horizon.md) | ![shipped](https://img.shields.io/badge/shipped-brightgreen) |
| [005 drto.dynamic_to_steady_state](005-drto.dynamic_to_steady_state.md) | ![implemented](https://img.shields.io/badge/implemented-yellowgreen) |
| [006 drto.dynamic_optimization](006-drto.dynamic_optimization.md) | ![ready](https://img.shields.io/badge/ready-blue) |
| [007 drto.dynamic_simulation](007-drto.dynamic_simulation.md) | ![ready](https://img.shields.io/badge/ready-blue) |
| [008 drto.steady_state_simulation](008-drto.steady_state_simulation.md) | ![implemented](https://img.shields.io/badge/implemented-yellowgreen) |
| [009 drto.steady_state_optimization](009-drto.steady_state_optimization.md) | ![ready](https://img.shields.io/badge/ready-blue) |
| [010 drto.initialize_steady_state](010-drto.initialize_steady_state.md) | ![implemented](https://img.shields.io/badge/implemented-yellowgreen) |
| [011 drto.cold_start_dynamic](011-drto.cold_start_dynamic.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |
| [012 drto.advanced_step_controller](012-drto.advanced_step_controller.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |
| [013 drto.warm_start_dynamic](013-drto.warm_start_dynamic.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |
| [014 drto.ideal_nmpc](014-drto.ideal_nmpc.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |
| [015 drto.asnmpc](015-drto.asnmpc.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |
| [016 drto.nonideal_nmpc](016-drto.nonideal_nmpc.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |
| [017 drto.parameterize](017-drto.parameterize.md) | ![shipped](https://img.shields.io/badge/shipped-brightgreen) |
| [018 Estimation declarations](018-estimation_declarations.md) | ![draft](https://img.shields.io/badge/draft-lightgrey) |

## Template

- `# <name>`: the capability or transform name, e.g. `drto.build_objective`.
- `**Status:**` at the top: the feature's current state, shown as one of the
  lifecycle badges above (`draft`, `ready`, `implemented`, `shipped`). Keep the
  status table above in sync.
- `## Description`: a user story, "As a user of DRTO, I want X, so that Y."
- `## Benefit hypothesis`: the value the feature is expected to deliver and
  why it is worth building.
- `## Acceptance criteria`: the concrete, testable conditions the
  implementation must satisfy. They drive the tests and the definition of
  done, which includes the feature's documentation (a `docs/` page and, where
  it applies, an example notebook).

`001-drto.info.md` is the first example.
