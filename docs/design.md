# Design

drto is built spec-first: every capability is specified as a feature
before it is implemented, and the specs are the authoritative statement
of what each piece does and why it is shaped that way. They live in the
repository under
[`features/`](https://github.com/devin-griff/drto/tree/main/features),
with a status table tracking each one through draft → ready →
implemented → shipped. The
[design record](https://github.com/devin-griff/drto/blob/main/DESIGN.md)
holds the cross-cutting decisions.

## What is real today

Shipped, in releases you can install: the registry and the declaration
surface, objective assembly, all six mode transformations (dynamic
optimization and simulation, the steady-state reduction, steady-state
simulation and optimization), the infinite-horizon terminal segment with
time-indexed Block support, control parameterization through pyomo-cvp,
steady-state initialization, cold start, registry units, and the
plotting.

Specified and ready, but **not yet implemented**: the advanced-step
controller (012), warm start (013), and the closed-loop frameworks —
ideal NMPC (014), advanced-step NMPC (015), and nonideal NMPC (016).
Until those land, drto builds, initializes, and solves open-loop
problems; the receding-horizon loop is on the roadmap, not in the
package.

## Why the specs are worth reading

Each spec records the user story, the benefit hypothesis, and testable
acceptance criteria, and the specs are amended as the design sharpens —
they are the best explanation of decisions like why the steady reduction
and the dynamic transforms are sibling branches of the same
declarations, why cold start interpolates instead of simulating forward,
or why the terminal segment's `beta` and `gamma` stay mutable. When the
docs and a spec disagree, the spec wins and the docs have a bug.
